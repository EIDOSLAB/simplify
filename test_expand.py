import time

import torch
from torch.nn.modules.conv import Conv2d
from torch.nn.utils import prune

from conv import ConvExpand


class Expander(Conv2d):
    def forward(self, x):
        x = super().forward(x)
        x = torch.cat([x, torch.zeros(x.shape[0], 1, *x.shape[2:]).to(x)], dim=1)
        
        return x[:, self.idxs]


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
    model = MockResidual()
    x = torch.randn(1, 3, 224, 224)
    
    for name, module in model.named_modules():
        if name == "conv_a_2" or name == "conv_b_1":
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')
    
    remaining_a = torch.nonzero(model.conv_a_2.weight.data.abs().sum(dim=(1, 2, 3)) != 0).view(-1)
    remaining_b = torch.nonzero(model.conv_b_1.weight.data.abs().sum(dim=(1, 2, 3)) != 0).view(-1)
    prune_a = torch.nonzero(~(model.conv_a_2.weight.data.abs().sum(dim=(1, 2, 3)) != 0)).view(-1)
    prune_b = torch.nonzero(~(model.conv_b_1.weight.data.abs().sum(dim=(1, 2, 3)) != 0)).view(-1)
    
    model.conv_a_2.bias.data[prune_a] = 0.
    model.conv_b_1.bias.data[prune_b] = 0.
    
    start_src = time.perf_counter()
    y_src = model(x)
    end_src = time.perf_counter() - start_src
    
    model.conv_a_2.weight.data = model.conv_a_2.weight.data[remaining_a]
    model.conv_a_2.bias.data = model.conv_a_2.bias.data[remaining_a]
    model.conv_b_1.weight.data = model.conv_b_1.weight.data[remaining_b]
    model.conv_b_1.bias.data = model.conv_b_1.bias.data[remaining_b]
    
    idxs = []
    current = 0
    for i in range(model.conv_a_2.weight.data.shape[0] + len(prune_a)):
        if i in prune_a:
            idxs.append(model.conv_a_2.weight.data.shape[0])
        else:
            idxs.append(current)
            current += 1
    
    model.conv_a_2 = ConvExpand.from_conv(model.conv_a_2, idxs)
    
    idxs = []
    current = 0
    for i in range(model.conv_b_1.weight.data.shape[0] + len(prune_b)):
        if i in prune_b:
            idxs.append(model.conv_b_1.weight.data.shape[0])
        else:
            idxs.append(current)
            current += 1
    
    model.conv_b_1 = ConvExpand.from_conv(model.conv_b_1, idxs)
    
    start_expand = time.perf_counter()
    y_expand = model(x)
    end_expand = time.perf_counter() - start_expand
    
    print(torch.allclose(y_src, y_expand))
    print("src time", end_src)
    print("expanded time", end_expand)
