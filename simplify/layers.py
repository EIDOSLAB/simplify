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
    def from_conv(module: nn.Conv2d, idxs: torch.Tensor, bias):
        module.__class__ = ConvExpand

        module.register_parameter('bf', torch.nn.Parameter(bias))
        setattr(module, "use_bf", bias.abs().sum() != 0)

        module.register_buffer('idxs', idxs.to(module.weight.device))
        module.register_buffer('zeros', torch.zeros(1, *bias.shape, dtype=bias.dtype, device=module.weight.device))

        setattr(module, 'idxs_cache', module.idxs)
        setattr(module, 'zero_cache', module.zeros)
        
        return module
    
    def forward(self, x):
        x = super().forward(x)
      
        zeros = self.zero_cache
        index = self.idxs_cache
        if zeros.shape[0] != x.shape[0]:
            zeros = self.zeros.expand(x.shape[0], *self.zeros.shape[1:])
            self.zero_cache = zeros

        if index.shape != x.shape:
            index = self.idxs[None, :, None, None].expand_as(x)
            self.idxs_cache = index

        expanded = torch.scatter(zeros, 1, index, x)
        return expanded + self.bf if self.use_bf else expanded
    
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
    def from_bn(module: nn.BatchNorm2d, idxs: torch.Tensor, bias, shape):
        module.__class__ = BatchNormExpand
        
        module.register_parameter('bf', torch.nn.Parameter(bias))

        module.register_buffer('idxs', idxs.to(module.weight.device))        
        module.register_buffer('zeros', torch.zeros(1, *shape[1:], dtype=bias.dtype, device=module.weight.device))

        setattr(module, 'zero_cache', module.zeros)
        setattr(module, 'idxs_cache', module.idxs)

        return module
    
    def forward(self, x):
        x = super().forward(x)

        zeros = self.zero_cache
        index = self.idxs_cache

        if zeros.shape[0] != x.shape[0]:
            zeros = self.zeros.expand(x.shape[0], *self.zeros.shape[1:])
            self.zero_cache = zeros

        if index.shape != x.shape:
            index = self.idxs[None, :, None, None].expand_as(x)
            self.idxs_cache = index

        expanded = torch.scatter(zeros, 1, index, x)        
        return expanded + self.bf[:, None, None].expand_as(expanded)
