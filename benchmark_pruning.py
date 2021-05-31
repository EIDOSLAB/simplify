import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from tabulate import tabulate
from torchvision.models.alexnet import alexnet
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_0, squeezenet1_1
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn

import profile
import simplify
import utils

device = 'cpu'


@torch.no_grad()
def run_pruning(architecture):
    print('\n----', architecture.__name__, '----')
    
    full_time, simplified_time = [], []
    model = architecture(pretrained=True).to(device)
    model.eval()
    
    for name, module in model.named_modules():
        if isinstance(model, SqueezeNet) and 'classifier.1' in name:
            continue
        
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')
    
    im = torch.randint(0, 256, (100, 3, 224, 224))
    x = (im / 255.).to(device)
    
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_src = model(x)
        full_time.append(time.perf_counter() - start)
    
    profiled = profile.profile_model(model, torch.randn((1, 3, 224, 224)), rows=1000)
    with open(f'profile/{architecture.__name__}.txt', 'w') as f:
        f.write('-- THRESHOLDED --\n')
        f.write(profiled)
    
    print('=> Full model inference time:', np.mean(full_time), np.std(full_time))
    
    pinned_out = utils.get_pinned_out(model)
    
    model = model.to('cpu')
    model = simplify.simplify(model, torch.zeros((1, 3, 224, 224)), pinned_out=pinned_out)
    model = model.to(device)
    
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_simplified = model(x)
        simplified_time.append(time.perf_counter() - start)
    
    profiled = profile.profile_model(model, torch.randn((1, 3, 224, 224)), rows=1000)
    with open(f'profile/{architecture.__name__}.txt', 'a') as f:
        f.write('\n\n -- SIMPLIFIED --\n')
        f.write(profiled)
    
    print('=> Simplified model inference time:', np.mean(simplified_time), np.std(simplified_time))
    print('Allclose logits:', torch.allclose(y_src, y_simplified))
    print('Equal predictions:', torch.equal(y_src.argmax(dim=1), y_simplified.argmax(dim=1)))
    print(
        f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_simplified.argmax(dim=1)).sum()}/{y_simplified.shape[0]}')
    
    return full_time, simplified_time


if __name__ == '__main__':
    utils.set_seed(3)
    
    table = []
    for architecture in [alexnet, resnet18, resnet34, resnet50, resnet101, resnet152, squeezenet1_0, squeezenet1_1,
                         vgg16, vgg16_bn, vgg19, vgg19_bn]:
        full_time, s_time = run_pruning(architecture)
        table.append([architecture.__name__, f'{np.mean(full_time):.4f}s±{np.std(full_time):.4f}',
                      f'{np.mean(s_time):.4f}s±{np.std(s_time):.4f}'])
    table = tabulate(table, headers=['Architecture', 'Pruned time', 'Simplified time (p=50%)'], tablefmt='github')
    print(table)
