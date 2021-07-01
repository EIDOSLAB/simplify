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
    def from_conv(module: nn.Conv2d, idxs, bias):
        module.__class__ = ConvExpand
        setattr(module, 'idxs', idxs)
        module.register_parameter('bf', torch.nn.Parameter(bias))
        
        shape = bias.shape
        module.register_buffer('zeros', torch.zeros(1, 1, *shape[1:], device=module.weight.device))

        return module
    
    def forward(self, x):
        x = super().forward(x)
        
        #zeros = self.zeros.repeat(x.shape[0], 1, 1, 1)
        #zeros = torch.zeros(x.shape[0], 1, *self.bf.shape[1:], device=self.weight.device)
        #x = torch.cat([x, zeros], dim=1)
        x = torch.nn.functional.pad(x, (0,0,0,0,0,1))
        return x[:, self.idxs] + self.bf
    
    def __repr__(self):
        return f'ConvExpand({self.in_channels}, {self.out_channels}, exp={len(self.idxs)})'


class BatchNormB(nn.BatchNorm2d):
    @staticmethod
    def from_bn(module: nn.BatchNorm2d, bias):
        module.__class__ = BatchNormB
        module.register_parameter('bf', torch.nn.Parameter(bias))
        return module
    
    def forward(self, x):
        x = super().forward(x)
        return x + self.bf[:, None, None].expand_as(x[0])

class BatchNormExpand(nn.BatchNorm2d):
    @staticmethod
    def from_bn(module: nn.BatchNorm2d, idxs, bias, shape):
        module.__class__ = BatchNormExpand
        setattr(module, 'idxs', idxs)

        module.register_parameter('bf', torch.nn.Parameter(bias))
        module.register_buffer('zeros', torch.zeros(1, 1, *shape[2:]))
        
        return module
    
    def forward(self, x):
        x = super().forward(x)
        
        zeros = torch.zeros(x.shape[0], 1, *self.bf.shape[1:], device=self.weight.device)
        #zeros = self.zeros.repeat(x.shape[0], 1, 1, 1)
        x = torch.cat([x, zeros], dim=1)
        x = x[:, self.idxs]
        
        return x + self.bf[:, None, None].expand_as(x[0])
