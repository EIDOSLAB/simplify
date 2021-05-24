import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

import fuser
import simplify


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


if __name__ == '__main__':
    random.seed(3)
    os.environ["PYTHONHASHSEED"] = str(3)
    np.random.seed(3)
    torch.cuda.manual_seed(3)
    torch.cuda.manual_seed_all(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(3)
    
    device = 'cpu'
    
    model = torchvision.models.resnet18(pretrained=True).to(device)
    model.eval()
    
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')
    
    x = torch.randn((32, 3, 224, 224)).to(device)
    
    total = []
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_src = model(x)
        total.append(time.perf_counter() - start)
    
    print('=> Full model inference time:', np.mean(total), np.std(total))
    # print(model)
    
    pinned_out = []
    if isinstance(model, ResNet):
        pinned_out = ['conv1']
        
        for name, module in model.named_modules():
            if isinstance(module, BasicBlock):
                pinned_out.append(f'{name}.conv2')
                if module.downsample is not None:
                    pinned_out.append(f'{name}.downsample.0')
            
            if isinstance(module, Bottleneck):
                pinned_out.append(f'{name}.conv3')
                if module.downsample is not None:
                    pinned_out.append(f'{name}.downsample.0')
    
    if isinstance(model, MockResidual):
        pinned_out = ["conv_a_2", "conv_b_1"]
    
    model = model.to('cpu')
    model = fuser.fuse(model)
    model = simplify.simplify(model, torch.randn((1, 3, 224, 224)), pinned_out=pinned_out)
    model = model.to(device)
    
    total = []
    for i in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            y_simplified = model(x)
        total.append(time.perf_counter() - start)
    
    print('=> Simplified model inference time:', np.mean(total), np.std(total))
    # print(model)
    print(torch.equal(y_src.argmax(dim=1), y_simplified.argmax(dim=1)))
