#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
from torch import nn
from torch.nn.utils import prune
from torchvision.models import SqueezeNet
from torchvision.models import wide_resnet101_2, resnet50, resnext101_32x8d, mobilenet_v3_large, mnasnet1_0, mnasnet1_3, \
    alexnet, densenet121, googlenet, inception_v3, resnet18, resnet34, resnet101, resnet152, densenet161, densenet169, \
    densenet201, mobilenet_v3_small, mobilenet_v2, resnext50_32x4d, wide_resnet50_2, mnasnet0_5, mnasnet0_75
from torchvision.models.shufflenetv2 import shufflenet_v2_x2_0, shufflenet_v2_x0_5, shufflenet_v2_x1_0, \
    shufflenet_v2_x1_5
from torchvision.models.squeezenet import squeezenet1_1, squeezenet1_0
from torchvision.models.vgg import vgg19_bn, vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19

models = [
    alexnet,
    resnet18,
    squeezenet1_0
]

# models = [
#     alexnet,
#     vgg11, vgg11_bn,
#     resnet18, resnet50,
#     squeezenet1_0,
#     densenet121,
#     inception_v3,
#     googlenet,
#     shufflenet_v2_x0_5,
#     mobilenet_v2, mobilenet_v3_small,
#     resnext50_32x4d,
#     wide_resnet50_2,
#     mnasnet0_5, mnasnet1_0,
#     densenet121
# ]


def get_model(architecture, arch):
    if architecture.__name__ in ["shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
        pretrained = False
    else:
        pretrained = True

    model = arch(pretrained, progress=False)
    model.eval()

    for name, module in model.named_modules():
        if isinstance(model, SqueezeNet) and 'classifier.1' in name:
            continue

        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            prune.random_structured(module, 'weight', amount=0.8, dim=0)
            prune.remove(module, 'weight')

    return model
