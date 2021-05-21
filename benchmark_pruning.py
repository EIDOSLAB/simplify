import torch
import torchvision
import torch.nn as nn
import torch.nn.utils.prune as prune
import time

from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet

import fuser
import simplify

model = torchvision.models.vgg16_bn(pretrained=True)
model.eval()

for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.random_structured(module, 'weight', amount=0.5, dim=0)
        prune.remove(module, 'weight')

x = torch.randn((32, 3, 224, 224))

start = time.perf_counter()
with torch.no_grad():
    y_src = model(x)
end = time.perf_counter()

print('=> Full model inference time:', end-start)

pinned_out = []
if isinstance(model, ResNet):
    pinned_out = ['conv1']

    for name, module in model.named_modules():
        if isinstance(module, BasicBlock):
            pinned_out.append(f'{name}.conv2')
            if module.downsample is not None:
                pinned_out.append(f'{name}.downsample.0')
        
        if isinstance(module, Bottleneck):
            pinned_out.append(f'{name}.conv3')
            if module.downsample is not None:
                pinned_out.append(f'{name}.downsample.0')
                
model = fuser.fuse(model)
model = simplify.simplify(model, torch.randn((1, 3, 224, 224)), pinned_out=pinned_out)

start = time.perf_counter()
with torch.no_grad():
    y_simplified = model(x)
end = time.perf_counter()

print('=> Simplified model inference time:', end-start)
print(torch.equal(y_src.argmax(dim=1), y_simplified.argmax(dim=1)))