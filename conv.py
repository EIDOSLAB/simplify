import torch
import torch.nn as nn

class ConvB(nn.Conv2d):
    @staticmethod
    def from_conv(module: nn.Conv2d, bias):
        module.__class__ = ConvB
        module.register_parameter('bf', torch.nn.Parameter(bias))
        return module
        
    def forward(self, x):
        x = super().forward(x)
        return x + self.bf