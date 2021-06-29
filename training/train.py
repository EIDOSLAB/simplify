import os
import random
import time

import numpy
import torch
import wandb
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.nn.utils import prune
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision.models import resnet50
from tqdm import tqdm

import utils
from simplify import propagate_bias, remove_zeroed
from training.data_loader_imagenet import get_data_loaders
from training.stats import architecture_stat
from training.test import test_model


def select_device(device=''):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability
    
    cuda = not cpu and torch.cuda.is_available()
    
    return torch.device('cuda:0' if cuda else 'cpu')


@torch.no_grad()
def apply_mask_neurons(model, mask):
    for n_m, mo in model.named_modules():
        if isinstance(mo, (nn.modules.Linear, nn.modules.Conv2d, nn.modules.ConvTranspose2d, nn.modules.BatchNorm2d)):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)
                if len(p.shape) == 1:
                    p.mul_(mask[name])
                elif len(p.shape) == 2:
                    p.copy_(torch.einsum(
                        'ij,i->ij',
                        p,
                        mask[name]
                    ))
                elif len(p.shape) == 4:
                    if isinstance(mo, nn.modules.Conv2d):
                        p.copy_(torch.einsum(
                            'ijnm,i->ijnm',
                            p,
                            mask[name]
                        ))
                    
                    if isinstance(mo, nn.modules.ConvTranspose2d):
                        p.copy_(torch.einsum(
                            'ijnm,j->ijnm',
                            p,
                            mask[name]
                        ))


@torch.no_grad()
def get_model_mask_neurons(model, layers):
    mask = {}
    for n_m, mo in model.named_modules():
        if isinstance(mo, layers):
            for n_p, p in mo.named_parameters():
                name = "{}.{}".format(n_m, n_p)
                
                if "weight" in n_p:
                    if isinstance(mo, nn.modules.Linear):
                        sum = torch.abs(p).sum(dim=1)
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    elif isinstance(mo, nn.modules.Conv2d):
                        sum = torch.abs(p).sum(dim=(1, 2, 3))
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    elif isinstance(mo, nn.modules.ConvTranspose2d):
                        sum = torch.abs(p).sum(dim=(0, 2, 3))
                        mask[name] = torch.where(sum == 0, torch.zeros_like(sum), torch.ones_like(sum))
                    else:
                        mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))
                else:
                    mask[name] = torch.where(p == 0, torch.zeros_like(p), torch.ones_like(p))
    
    return mask


def get_masks(model):
    mask_neurons = get_model_mask_neurons(model, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))
    return mask_neurons


def train(model, train_loader, test_loader, pytorch_optimizer, wandb_writer, device):
    epochs = 20
    steps = list(numpy.arange(10, epochs, 10))
    lr_scheduler = MultiStepLR(pytorch_optimizer, steps, gamma=0.1)
    loss_function = CrossEntropyLoss()
    task = "classification"
    
    scaler = GradScaler()
    
    zeros = torch.zeros(1, 3, 224, 224).to(device)
    amount = 0
    
    # Epochs
    for epoch in range(epochs):
        
        model.train()
        
        # Epoch progress bar
        print("")
        epoch_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training epoch {}".format(epoch))
        
        mask_neurons = get_masks(model)
        
        times = []
        
        # Batches
        for batch, (data, target) in epoch_pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            pytorch_optimizer.zero_grad()
            
            with autocast():
                start = time.time()
                output = model(data)
                times.append(time.time() - start)
                loss = loss_function(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(pytorch_optimizer)
            scaler.update()
            
            apply_mask_neurons(model, mask_neurons)
        
        amount += 0.05
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
                prune.remove(module, 'weight')
        
        # model.eval()
        # pinned_out = utils.get_pinned_out(model)
        # propagate_bias(model, zeros, pinned_out)
        # remove_zeroed(model, zeros, pinned_out)
        # model.train()
        
        # Test this epoch model
        eval_epoch(epoch, model, loss_function, test_loader, device, task,
                   pytorch_optimizer, wandb_writer, numpy.mean(times))
        
        lr_scheduler.step()


def eval_epoch(epoch, model, loss_function, test_loader, device, task, pytorch_optimizer, wandb_writer, epoch_time):
    test_performance = test_model(model, loss_function, test_loader, device, task, desc="Evaluating model on test set")
    
    lr = [p['lr'] for p in pytorch_optimizer.param_groups]
    wd = [p['weight_decay'] for p in pytorch_optimizer.param_groups]
    mom = [p['momentum'] for p in pytorch_optimizer.param_groups]
    
    wandb_writer.log(
        {"Performance/Test/{}".format("Top-1" if task == "classification" else "Jaccard"): test_performance[0]}, step=epoch)
    wandb_writer.log(
        {"Performance/Test/{}".format("Top-5" if task == "classification" else "Dice"): test_performance[1]}, step=epoch)
    wandb_writer.log({"Performance/Test/Loss": test_performance[2]}, step=epoch)
    for i, val in enumerate(lr):
        wandb_writer.log({"Params/Learning Rate {}".format(i): val}, step=epoch)
    for i, val in enumerate(wd):
        wandb_writer.log({"Params/Weight Decay {}".format(i): val}, step=epoch)
    for i, val in enumerate(mom):
        wandb_writer.log({"Params/Momentum {}".format(i): val}, step=epoch)
    
    pruning_stat = architecture_stat(model)
    wandb_writer.log({"Architecture/Neurons Percentage": pruning_stat["network_neuron_non_zero_perc"]}, step=epoch)
    wandb_writer.log({"Architecture/Neurons CR": pruning_stat["network_neuron_ratio"]}, step=epoch)
    wandb_writer.log({"Architecture/Parameters Percentage": pruning_stat["network_param_non_zero_perc"]}, step=epoch)
    wandb_writer.log({"Architecture/Parameters CR": pruning_stat["network_param_ratio"]}, step=epoch)
    
    wandb_writer.log({"Epoch Time": epoch_time}, step=epoch)


if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    numpy.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = select_device("0")
    model = resnet50(False).to(device)
    
    train_loader, test_loader = get_data_loaders("/data01/ImageNet", 128, 128, 0, True, 8, False)
    pytorch_optimizer = SGD(model.parameters(), lr=0.1, momentum=0, weight_decay=0)
    
    wdb_writer = wandb.init(project="Simplification Train", name="Baseline")
    
    train(model, train_loader, test_loader, pytorch_optimizer, wdb_writer, device)
    
    wdb_writer.finish()
