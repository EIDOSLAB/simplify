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


class ConvExpand(ConvB):
    
    @staticmethod
    def from_conv(module: ConvB, idxs):
        module.__class__ = ConvExpand
        setattr(module, 'idxs', idxs)
        return module
    
    def forward(self, x):
        x = super().forward(x)
        # TODO move the torch.zeros to __init__
        x = torch.cat([x, torch.zeros(x.shape[0], 1, *x.shape[2:]).to(x)], dim=1)
        
        return x[:, self.idxs]
