import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils import prune
from tqdm import tqdm

import os
import simplify
from tests.benchmark_models import models

device = torch.device(os.environ.get('DEVICE', 'cuda'))


def time_model(model, x, y):
    forward_time = []
    backward_time = []
    model.to(device)
    for j in range(10):  # tqdm(range(10), desc="Pruned test"):
        if device == torch.device("cuda"):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
        else:
            start = time.perf_counter()
        
        output = model(x)  # FORWARD PASS
        
        if device == torch.device("cuda"):
            ender.record()
            torch.cuda.synchronize()
        else:
            end = time.perf_counter()
        
        forward_time.append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
        
        loss = F.cross_entropy(output, y)
        
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
        
        backward_time.append(starter.elapsed_time(ender) if device == torch.device("cuda") else end - start)
    
    forward_time = forward_time[1:]
    backward_time = backward_time[1:]
    
    return forward_time, backward_time


def main(network):
    print('=> Benchmarking', network.__name__)
    model = network(False)
    
    batch_size = 128
    fake_input = torch.randint(0, 256, (batch_size, 3, 224, 224))
    fake_input = fake_input.float() / 255.
    fake_target = torch.randint(0, 1000, (batch_size,)).long()
    
    fake_input, fake_target = fake_input.to(device), fake_target.to(device)
    
    criterion = CrossEntropyLoss()
    
    prune_step = 0.05
    iterations = int(1. / prune_step)
    
    total_neurons = 0
    remaining_neurons = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            total_neurons += module.weight.shape[0]
            remaining_neurons += module.weight.shape[0]
    
    x = []
    pruned_y_forward = []
    pruned_y_forward_std = []
    pruned_y_backward = []
    pruned_y_backward_std = []
    
    simplified_y_forward = []
    simplified_y_forward_std = []
    simplified_y_backward = []
    simplified_y_backward_std = []
    
    amount = 0.
    
    wandb.init(
        config={'arch': network.__name__},
        group="benchmark_passes"
    )
    
    for i in tqdm(range(iterations), desc="Benchmark"):
        if amount > 1.:
            break
        model = network(False)
        # simplify.fuse(model, simplify.utils.get_bn_folding(model))
        
        # First loop is the full model
        if i > 0:
            remaining_neurons = 0
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, 'weight', amount=amount, n=2, dim=0)
                    ch_sum = module.weight.sum(dim=(1, 2, 3))
                    remaining_neurons += ch_sum[ch_sum != 0].shape[0]
                    prune.remove(module, 'weight')
        
        x.append(100 - (remaining_neurons / total_neurons) * 100)
        
        # PRUNED
        forward_time, backward_time = time_model(model, fake_input, fake_target)
        
        pruned_y_forward.append(np.mean(forward_time))
        pruned_y_forward_std.append(np.std(forward_time))
        pruned_y_backward.append(np.mean(backward_time))
        pruned_y_backward_std.append(np.std(backward_time))
        
        wandb.log({
            'pruned.forward':      np.mean(forward_time),
            'pruned.forward_std':  np.std(forward_time),
            'pruned.backward':     np.mean(backward_time),
            'pruned.backward_std': np.std(backward_time)
        }, commit=False)
        
        # SIMPLIFIED
        model.eval()
        simplify.simplify(model, torch.zeros(1, 3, 224, 224).to(device), fuse_bn=False, training=True)
        model.train()
        
        forward_time, backward_time = time_model(model, fake_input, fake_target)
        
        simplified_y_forward.append(np.mean(forward_time))
        simplified_y_forward_std.append(np.std(forward_time))
        simplified_y_backward.append(np.mean(backward_time))
        simplified_y_backward_std.append(np.std(backward_time))
        
        wandb.log({
            'simplified.forward':      np.mean(forward_time),
            'simplified.forward_std':  np.std(forward_time),
            'simplified.backward':     np.mean(backward_time),
            'simplified.backward_std': np.std(backward_time),
            'remaining_neurons':       100. - x[-1]
        })
        
        amount += prune_step
        del model
        torch.cuda.empty_cache()
    
    wandb.finish()


if __name__ == '__main__':
    for arch in models:
        main(arch)
