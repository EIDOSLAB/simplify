import argparse
import time

import torch
import wandb
from torch import nn
from torch.autograd import profiler
from torch.nn import CrossEntropyLoss
from torch.nn.utils import prune
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50
from tqdm import tqdm

import simplify
from simplify.utils import set_seed


def profile_model(model, input, rows=10, cuda=False):
    with profiler.profile(profile_memory=True, record_shapes=True, use_cuda=cuda) as prof:
        with profiler.record_function("model_inference"):
            model(input)
    
    return str(prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=rows))


def prune_model(model, amount, remove=False):
    print("=> Pruning")
    
    remaining_neurons = 0.
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, 'weight', amount=amount, n=2, dim=0)
            w = module.weight.clone().reshape(module.weight.shape[0], -1).abs().sum(dim=1)
            remaining_neurons += (w != 0).sum()
            
            if remove:
                prune.remove(module, 'weight')
    
    return remaining_neurons


def main(config):
    set_seed(0)
    device = torch.device('cuda')
    
    batch_size = 128
    train_iteration = 10000
    prune_iteration = config.prune_every
    
    model = resnet50(False).to(device)
    simplify.fuse(model, simplify.utils.get_bn_folding(model))
    
    optimizer = SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, train_iteration, 1e-3)
    criterion = CrossEntropyLoss()
    
    total_neurons = 0
    remaining_neurons = 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            total_neurons += module.weight.shape[0]
            remaining_neurons += module.weight.shape[0]
    
    wandb.init(config=config)
    
    model.eval()
    profiled = profile_model(model, torch.randn((batch_size, 3, 224, 224), device=device), rows=1000)
    with open('profile.txt', 'w') as f:
        f.write('\n\n -- THRESHOLDED --\n')
        f.write(profiled)
    
    # Train
    num_samples = 0
    num_correct = 0
    
    for i in tqdm(range(train_iteration)):
        images = torch.randn((batch_size, 3, 224, 224), device=device)
        target = torch.randint(0, 1000, (batch_size,), device=device)
        
        # Prune the network by 5% at each pass
        if (i + 1) % prune_iteration == 0:
            remaining_neurons = prune_model(model, amount=0.10, remove=config.simplify)
            print(f"The current model has {(remaining_neurons / total_neurons) * 100} % of the original neurons")
            
            if config.simplify:
                print("Simplifying model")
                model.eval()
                simplify.simplify(model, torch.zeros(1, 3, 224, 224, device=device), fuse_bn=False, training=True)
                model.train()
                
                profiled = profile_model(model, torch.randn((batch_size, 3, 224, 224), device=device), rows=1000)
                with open('profile.txt', 'a') as f:
                    f.write(f'\n\n -- SIMPLIFIED {(remaining_neurons / total_neurons) * 100} --\n')
                    f.write(profiled)
                # torch.cuda.empty_cache()
                
                # Re-init optimizer and scheduler
                optimizer = SGD(model.parameters(), lr=0.1, weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, train_iteration, 1e-3, last_epoch=-1)
                for _ in range(i):
                    scheduler.step()
        
        model.train()
        with torch.enable_grad():
            
            if device == torch.device("cuda"):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
            else:
                start = time.perf_counter()
            
            output = model(images)  # FORWARD PASS
            
            if device == torch.device("cuda"):
                ender.record()
                torch.cuda.synchronize()
            else:
                end = time.perf_counter()

            forward_time = starter.elapsed_time(ender) if device == torch.device("cuda") else end - start
            
            loss = criterion(output, target)
            optimizer.zero_grad()
            
            if device == torch.device("cuda"):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
            else:
                start = time.perf_counter()
                
            loss.backward()  # BACKWARD PASS
            
            if device == torch.device("cuda"):
                ender.record()
                torch.cuda.synchronize()
            else:
                end = time.perf_counter()

            backward_time = starter.elapsed_time(ender) if device == torch.device("cuda") else end - start
        
        for param in model.parameters():
            param.grad.data.mul_(torch.abs(param.data) > 0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        _, predictions = output.max(1)
        num_correct += (predictions == target).sum()
        num_samples += predictions.size(0)
        
        to_log = {
            "Remaining neurons": (remaining_neurons / total_neurons),
            "Train Accuracy":    float(num_correct) / float(num_samples),
            "Forward Time":      forward_time,
            "Backward Time":     backward_time,
            "epoch":             i
        }
        
        current_lr = [group["lr"] for group in optimizer.param_groups]
        
        for j, lr in enumerate(current_lr):
            to_log[f"lr{j}"] = lr
        
        wandb.log(to_log)
        
        if (i + 1) % train_iteration == 0:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_every', type=int, default=1000)
    parser.add_argument('--simplify', action='store_true')
    config = parser.parse_args()
    main(config)
