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
        # module.register_parameter('zeros', torch.nn.Parameter(torch.zeros(shape[0], 1, *shape[2:])))
        return module
    
    def forward(self, x):
        x = super().forward(x)
        # TODO mode zeros to from_conv
        zeros = torch.zeros(x.shape[0], 1, *x.shape[2:]).to(x)
        x = torch.cat([x, zeros], dim=1)
        self.bf.data = torch.cat([self.bf.data, zeros[0]], dim=0)
        
        return x[:, self.idxs] + self.bf[self.idxs]
