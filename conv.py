import torch
import torch.nn as nn
from torch.nn.modules.conv import Conv2d

class ConvB(Conv2d):
    def forward(self, x):
        x = super().forward(x)
        return x + self.bf