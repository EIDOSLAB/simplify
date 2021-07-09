import torch
import torch.nn as nn


# TODO there are many at run-time operations, too many
from torch.nn.functional import pad


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

        select_idxs = []
        current = 0
        for i in range(bias.shape[0]):
            if i in idxs:
                select_idxs.append(current)
                current += 1
            else:
                select_idxs.append(module.weight.shape[0])
        select_idxs = torch.tensor(select_idxs)
        
        module.register_buffer('idxs', idxs.to(module.weight.device))
        module.register_buffer('select_idxs', select_idxs.to(module.weight.device))

        module.register_parameter('bf', torch.nn.Parameter(bias))
        setattr(module, "use_bf", bias.abs().sum() != 0)

        module.register_buffer('zeros', torch.zeros(1, *bias.shape, dtype=bias.dtype, device=module.weight.device))
        setattr(module, 'zero_cache', module.zeros)
        setattr(module, 'idxs_cache', module.idxs)
        
        return module
    
    def forward(self, x):
        x = super().forward(x)
        # x = pad(x, (0, 0, 0, 0, 0, 1))
        # expanded = x[:, self.select_idxs]

        # idxs = self.select_idxs[None, :, None, None].expand(x.shape[0], self.select_idxs.shape[0], *x.shape[2:])
        # expanded = torch.gather(x, dim=1, index=idxs)
        
        zeros = self.zero_cache
        index = self.idxs_cache
        if zeros.shape[0] != x.shape[0]:
            zeros = self.zeros.expand(x.shape[0], *self.zeros.shape[1:])
            index = self.idxs[None, :, None, None].expand_as(x)
            self.zero_cache = zeros
            self.idxs_cache = index
        expanded = torch.scatter(zeros, 1, index, x)

        # expanded = torch.index_select(x, 1, self.select_idxs)
        
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

        select_idxs = []
        current = 0
        for i in range(bias.shape[0]):
            if i in idxs:
                select_idxs.append(current)
                current += 1
            else:
                select_idxs.append(module.weight.shape[0])
        select_idxs = torch.tensor(select_idxs)
        
        module.register_buffer('idxs', idxs.to(module.weight.device))
        module.register_buffer('select_idxs', select_idxs.to(module.weight.device))
        module.register_parameter('bf', torch.nn.Parameter(bias))
        
        module.register_buffer('zeros', torch.zeros(1, 1, *shape[2:], dtype=bias.dtype, device=module.weight.device))
        setattr(module, 'zero_cache', module.zeros)
        setattr(module, 'idxs_cache', module.idxs)

        return module
    
    def forward(self, x):
        x = super().forward(x)
        # x = pad(x, (0, 0, 0, 0, 0, 1))
        # expanded = x[:, self.select_idxs]

        # idxs = self.select_idxs[None, :, None, None].expand(x.shape[0], self.select_idxs.shape[0], *x.shape[2:])
        # expanded = torch.gather(x, dim=1, index=idxs)
        
        zeros = self.zero_cache
        index = self.idxs_cache
        if zeros.shape[0] != x.shape[0]:
            zeros = self.zeros.expand(x.shape[0], self.bf.shape[0], *self.zeros.shape[2:])
            index = self.idxs[None, :, None, None].expand_as(x)
            self.zero_cache = zeros
            self.idxs_cache = index
        expanded = torch.scatter(zeros, 1, index, x)

        # expanded = torch.index_select(x, 1, self.select_idxs)
        
        return expanded + self.bf[:, None, None].expand_as(expanded)
