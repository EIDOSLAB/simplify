import torch
import torch.nn as nn


# TODO there are many at run-time operations, too many
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
        
        module.register_buffer('idxs', idxs.to(module.weight.device))
        module.register_parameter('bf', torch.nn.Parameter(bias))
        module.register_buffer('zeros', torch.zeros(1, *bias.shape, dtype=bias.dtype, device=module.weight.device))

        return module
    
    def forward(self, x):
        x = super().forward(x)
        
        zeros = self.zeros.expand(x.shape[0], *self.zeros.shape[1:])
        index = self.idxs[None, :, None, None].expand_as(x)
        expanded = torch.scatter(zeros, 1, index, x)
        
        return expanded + self.bf
    
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
        
        module.register_buffer('idxs', idxs.to(module.weight.device))
        module.register_parameter('bf', torch.nn.Parameter(bias))
        module.register_buffer('zeros', torch.zeros(1, 1, *shape[2:], dtype=bias.dtype, device=module.weight.device))
        
        return module
    
    def forward(self, x):
        x = super().forward(x)
        
        # zeros = torch.zeros(x.shape[0], self.bf.shape[0], *x.shape[2:], dtype=x.dtype, device=self.weight.device)
        zeros = self.zeros.expand(x.shape[0], self.bf.shape[0], *self.zeros.shape[2:])
        index = self.idxs[None, :, None, None].expand_as(x)
        expanded = torch.scatter(zeros, 1, index, x)
        
        return expanded + self.bf[:, None, None].expand_as(expanded)
