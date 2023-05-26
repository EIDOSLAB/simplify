#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
import os
import random

import numpy as np
import torch.onnx
from torch import nn
from torch._C._onnx import TrainingMode
from torch.nn.utils import prune
from torchvision.models import SqueezeNet
from torchvision.models import alexnet, resnet18
from torchvision.models.squeezenet import squeezenet1_0


class ResidualNet(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2, 10)

        self.init_bn(self.bn1)
        self.init_bn(self.bn2)

    def init_bn(self, bn):
        bn.weight.data = torch.randn_like(bn.weight.data)
        bn.bias.data = torch.randn_like(bn.bias.data)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = self.conv2(x)
        x2 = self.bn2(x2)

        x3 = x + x2
        x3 = self.relu(x3)

        x4 = self.avgpool(x3)
        x4 = torch.flatten(x4, 1)
        x4 = self.fc(x4)

        return x4


# models = [
#     ResidualNet
# ]

# models = [
#     alexnet,
#     resnet18,
#     squeezenet1_0
# ]


models = [
    alexnet,
    vgg11, vgg11_bn,
    resnet18, resnet50,
    squeezenet1_0,
    densenet121,
    inception_v3,
    googlenet,
    shufflenet_v2_x0_5,
    mobilenet_v2, mobilenet_v3_small,
    resnext50_32x4d,
    wide_resnet50_2,
    mnasnet0_5, mnasnet1_0,
    densenet121
]


def get_model(architecture, arch):
    # random.seed(0)
    # os.environ["PYTHONHASHSEED"] = str(0)
    # np.random.seed(0)
    # torch.cuda.manual_seed(0)
    # torch.cuda.manual_seed_all(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.manual_seed(0)

    if architecture.__name__ in ["shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
        pretrained = False
    else:
        pretrained = True

    model = arch(pretrained, progress=False)
    model(torch.randn(64, 3, 224, 224))
    model.eval()

    for name, module in model.named_modules():
        if isinstance(model, SqueezeNet) and 'classifier.1' in name:
            continue

        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')

        # if isinstance(module, nn.BatchNorm2d):
        #     prune.random_unstructured(module, 'weight', amount=0.5)
        #     prune.remove(module, 'weight')

    return model


if __name__ == '__main__':
    model = ResidualNet()
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), "resnet18.onnx", training=TrainingMode.TRAINING)
