import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils import prune
from torchvision.models import resnet18
from tqdm import tqdm

from simplify import propagate, remove_zeroed
from simplify.utils import get_pinned_out

if __name__ == '__main__':
    model = resnet18(True)
    device = torch.device("cuda")
    fake_input = torch.randint(0, 256, (256, 3, 224, 224))
    fake_input = fake_input.float() / 255.
    fake_target = torch.randint(0, 1000, (256,)).long()
    
    fake_input, fake_target = fake_input.to(device), fake_target.to(device)
    
    criterion = CrossEntropyLoss()
    
    iterations = 2
    
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
    for i in tqdm(range(iterations), desc="Benchmark"):
        model = resnet18(True)
        
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
        forward_time = []
        backward_time = []
        model.to(device)
        for j in tqdm(range(10), desc="Pruned test"):
            start = time.perf_counter()
            output = model(fake_input)
            forward_time.append(time.perf_counter() - start)
            
            loss = criterion(output, fake_target)
            start = time.perf_counter()
            loss.backward()
            backward_time.append(time.perf_counter() - start)
        
        pruned_y_forward.append(np.mean(forward_time))
        pruned_y_forward_std.append(np.std(forward_time))
        pruned_y_backward.append(np.mean(backward_time))
        pruned_y_backward_std.append(np.std(backward_time))
        
        # SIMPLIFIED
        pinned_out = get_pinned_out(model)
        propagate.propagate_bias(model, torch.zeros(1, 3, 224, 224).to(device), pinned_out)
        remove_zeroed(model, torch.zeros(1, 3, 224, 224).to(device), pinned_out)
        
        forward_time = []
        backward_time = []
        model.to(device)
        for j in tqdm(range(10), desc="Simplified test"):
            start = time.perf_counter()
            output = model(fake_input)
            forward_time.append(time.perf_counter() - start)
            
            loss = criterion(output, fake_target)
            start = time.perf_counter()
            loss.backward()
            backward_time.append(time.perf_counter() - start)
        
        simplified_y_forward.append(np.mean(forward_time))
        simplified_y_forward_std.append(np.std(forward_time))
        simplified_y_backward.append(np.mean(backward_time))
        simplified_y_backward_std.append(np.std(backward_time))
        
        amount += 0.05
    
    plt.errorbar(x, pruned_y_forward, pruned_y_forward_std, label="pruned")
    plt.errorbar(x, simplified_y_forward, simplified_y_forward_std, label="simplified")
    plt.title("Forward")
    plt.xlabel("Pruning (%)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("Forward.png", dpi=300)
    plt.clf()
    
    plt.errorbar(x, pruned_y_backward, pruned_y_backward_std, label="pruned")
    plt.errorbar(x, simplified_y_backward, simplified_y_backward_std, label="simplified")
    plt.title("Backward")
    plt.xlabel("Pruning (%)")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.savefig("Backward.png", dpi=300)
    plt.clf()
