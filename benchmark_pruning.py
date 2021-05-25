import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torchvision.models import alexnet
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.models.resnet import *
from torchvision.models.vgg import *

import fuser
import simplify
import utils

from tabulate import tabulate

class MockResidual(torch.nn.Module):
    def __init__(self):
        super(MockResidual, self).__init__()
        self.conv_a_1 = torch.nn.Conv2d(3, 10, 5)
        self.conv_a_2 = torch.nn.Conv2d(10, 10, 5)
        self.conv_b_1 = torch.nn.Conv2d(3, 10, 9)
        self.conv_c_1 = torch.nn.Conv2d(10, 1, 5)
        
        self.linear = torch.nn.Linear(44944, 10)
    
    def forward(self, x):
        out_a = self.conv_a_1(x)
        out_a = self.conv_a_2(out_a)
        
        out_b = self.conv_b_1(x)
        
        out_a_b = out_a + out_b
        
        out_c = self.conv_c_1(out_a_b)
        
        out_lin = self.linear(out_c.view(out_c.shape[0], -1))
        return out_lin

device = 'cpu'

def run_pruning(architecture):
    print('\n----', architecture.__name__, '----')

    full_time, simplified_time = [], []
    model = architecture(pretrained=True).to(device)
    model.eval()
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')
    
    x = torch.randn((100, 3, 224, 224)).to(device)
    
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_src = model(x)
        full_time.append(time.perf_counter() - start)
    
    print('=> Full model inference time:', np.mean(full_time), np.std(full_time))
    
    pinned_out = utils.get_pinned_out(model)
    if isinstance(model, MockResidual):
        pinned_out = ["conv_a_2", "conv_b_1"]
    
    model = model.to('cpu')
    model = fuser.fuse(model)
    model = simplify.simplify(model, torch.randn((1, 3, 224, 224)), pinned_out=pinned_out)
    model = model.to(device)
    
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_simplified = model(x)
        simplified_time.append(time.perf_counter() - start)
    
    print('=> Simplified model inference time:', np.mean(simplified_time), np.std(simplified_time))
    print('Allclose logits:', torch.allclose(y_src, y_simplified))
    print('Equal predictions:', torch.equal(y_src.argmax(dim=1), y_simplified.argmax(dim=1)))
    print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_simplified.argmax(dim=1)).sum()}/{y_simplified.shape[0]}')

    return full_time, simplified_time

if __name__ == '__main__':
    random.seed(3)
    os.environ["PYTHONHASHSEED"] = str(3)
    np.random.seed(3)
    torch.cuda.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(3)
    torch.set_default_dtype(torch.float64)

    table = []
    for architecture in [alexnet, resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg16_bn, vgg19, vgg19_bn]:
        full_time, s_time = run_pruning(architecture)
        table.append([architecture.__name__, f'{np.mean(full_time):.4f}s±{np.std(full_time):.4f}', f'{np.mean(s_time):.4f}s±{np.std(s_time):.4f}'])
    table = tabulate(table, headers=['Architecture', 'Pruned time', 'Simplified time (p=50%)'], tablefmt='github')
    print(table)



