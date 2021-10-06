from torchvision.models import wide_resnet101_2, resnet50, resnext101_32x8d, mobilenet_v3_large, mnasnet1_0, alexnet, \
    densenet121, googlenet, inception_v3
from torchvision.models.shufflenetv2 import shufflenet_v2_x2_0
from torchvision.models.squeezenet import squeezenet1_1
from torchvision.models.vgg import vgg19_bn

from simplify.utils import get_pinned

models = [
    # alexnet,
    # vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
    # resnet18, resnet34, resnet50, resnet101, resnet152,
    # squeezenet1_0, squeezenet1_1,
    # densenet121, densenet161, densenet169, densenet201,
    # inception_v3,
    # googlenet,
    # shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,
    # mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large,
    # resnext50_32x4d, resnext101_32x8d,
    # wide_resnet50_2, wide_resnet101_2,
    # mnasnet0_5, mnasnet0_75, mnasnet1_0, nascent1_3,
    # densenet121,
    
    alexnet,
    densenet121,
    googlenet,
    inception_v3,
    mnasnet1_0,
    mobilenet_v3_large,
    resnet50,
    resnext101_32x8d,
    shufflenet_v2_x2_0,
    squeezenet1_1,
    vgg19_bn,
    wide_resnet101_2
]

if __name__ == '__main__':
    model = resnext101_32x8d()
    pinned = get_pinned(model)
    print()