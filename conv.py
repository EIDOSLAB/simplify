import torch
import torch.nn as nn

class ConvB(nn.Module):
    def __init__(self, conv, bf):
        super().__init__()
        self.conv = conv
        self.bf = bf

    def forward(self, x):
        return self.conv(x) + self.bf