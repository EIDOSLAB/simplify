from simplify.fuse import fuse
import torch
import torch.nn as nn
import simplify
import torch.nn.utils.prune as prune
import torchvision

model = nn.Sequential(
    nn.Conv2d(3, 10, 3, 1, 1, bias=False),
    nn.ReLU(),
    nn.Conv2d(10, 10, 3, 1, 1, bias=False),
    nn.ReLU(),
    nn.Conv2d(10, 10, 3, 1, 1, bias=False),
    nn.ReLU(),
)

model = torchvision.models.resnet18(True)

for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.random_structured(module, 'weight', amount=0.8, dim=0)
        prune.remove(module, 'weight')

print('=> Original model:', model)

simplify.simplify(model, torch.ones((1, 3, 224, 224)), fuse_bn=False, training=True)

#simplify.propagate_bias(model, torch.ones((1, 3, 128, 128)), [])
#simplify.simplify(model, torch.randn((1, 3, 128, 128)))
print('=> Propagated:', model)
