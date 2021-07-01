import random
import time

import numpy
import torch
import wandb
import argparse
import os

from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils import prune
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50

from simplify import fuse
from simplify.utils import get_bn_folding
from training.data_loader_imagenet import get_data_loaders
from tqdm import tqdm


def test(loader, model, device='cuda'):
    num_correct = 0
    num_samples = 0
    
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)
        
        return float(num_correct) / float(num_samples)


def main(config):
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    numpy.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device('cuda')

    model = resnet50(False).to(device)
    #bn_folding = get_bn_folding(model)
    #model = fuse(model, bn_folding)
    train_iteration = 10000
    prune_iteration = 1000
    test_iteration  = 1000
    
    train_loader, test_loader = get_data_loaders(os.path.join(config.root, "ImageNet"), 256, 256, 0, True, 8, False)
    optimizer = SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    scheduler = CosineAnnealingLR(optimizer, train_iteration, 1e-3)
    criterion = CrossEntropyLoss()
    
    total_neurons = 0
    remaining_neurons = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total_neurons += module.weight.shape[0]
            remaining_neurons += module.weight.shape[0]
    
    wandb.init()
    wandb.watch(model)
    
    # warmup
    # model(torch.randn(256, 3, 224, 224, device=device))

    # Train
    num_samples = 0
    num_correct = 0

    for i, (images, target) in enumerate(tqdm(train_loader)):
        
        model.train()
        images, target = images.to(device), target.to(device)

        # Prune the network by 5% at each pass
        if (i + 1) % prune_iteration == 0:
            print("Pruning")
            
            remaining_neurons = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, 'weight', amount=0.05, n=2, dim=0)
                    ch_sum = module.weight.sum(dim=(1, 2, 3))
                    remaining_neurons += ch_sum[ch_sum != 0].shape[0]
            
            print(f"The current model has {(remaining_neurons / total_neurons) * 100} % of the original neurons")
            
            optimizer = SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, train_iteration, 1e-3, last_epoch=-1)
            for _ in range(i):
                scheduler.step()
        
        with torch.cuda.amp.autocast():
            start = time.time()
            output = model(images)
            forward_time = time.time() - start
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            
            start = time.time()
            scaler.scale(loss).backward()
            backward_time = time.time() - start
            
            _, predictions = output.max(1)
            num_correct += (predictions == target).sum()
            num_samples += predictions.size(0)

        # Set to 0 the gradient of pruned neurons
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    for n, p in module.named_parameters():
                        if n == "weight_orig":
                            p.grad.mul_(module.weight_mask)
                        if n == "bias":
                            p.grad.mul_(module.weight_mask[:, 0, 0, 0])
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        
        to_log = {
            "Remaining neurons": (remaining_neurons / total_neurons),
            "Train Accuracy":  float(num_correct) / float(num_samples),
            "Forward Time": forward_time,
            "Backward Time": backward_time,
            "epoch": i
        }
        
        if (i + 1) % test_iteration == 0:
            # Test
            model.eval()
            accuracy = test(test_loader, model)
            to_log["Test Accuracy"] = accuracy

        current_lr = [group["lr"] for group in optimizer.param_groups]
        
        for j, lr in enumerate(current_lr):
            to_log[f"lr{j}"] = lr
        
        wandb.log(to_log)
        
        if (i + 1) % train_iteration == 0:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default=f'{os.path.expanduser("~")}/data')
    config = parser.parse_args()
    main(config)