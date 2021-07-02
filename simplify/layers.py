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

        module.register_buffer(
            'idxs', torch.tensor(
                idxs, device=module.weight.device))
        module.register_parameter('bf', torch.nn.Parameter(bias))

        return module

    def forward(self, x):
        x = super().forward(x)

        # Add zeroed channel
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 1))
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

        module.register_buffer(
            'idxs', torch.tensor(
                idxs, device=module.weight.device))
        module.register_parameter('bf', torch.nn.Parameter(bias))

        return module

    def forward(self, x):
        x = super().forward(x)

        # Add zeroed channel
        x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, 1))
        x = x[:, self.idxs]

        return x + self.bf[:, None, None].expand_as(x[0])
