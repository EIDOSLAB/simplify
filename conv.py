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


class ConvExpand(nn.Conv2d):
    
    @staticmethod
    def from_conv(module: ConvB, idxs, bias):
        module.__class__ = ConvExpand
        setattr(module, 'idxs', idxs)
        module.register_parameter('bf', torch.nn.Parameter(bias))

        shape = bias.shape
        module.register_buffer('zeros', torch.zeros(1, 1, *shape[2:]))

        return module
    
    def forward(self, x):
        x = super().forward(x)

        zeros = self.zeros.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat([x, zeros], dim=1)
        
        return x[:, self.idxs] + self.bf
