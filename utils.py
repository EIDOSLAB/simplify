import random
import os
import torch
import numpy as np

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def get_pinned_out(model):
    pinned_out = []

    if isinstance(model, ResNet):
        pinned_out = {'conv1': []}
        last_module = [('conv1', model.conv1)]

        all_modules = []
        for name, module in model.named_modules():
            if not isinstance(module, (BasicBlock, Bottleneck)):
                continue

            if module.downsample is not None:
                all_modules.append(module.downsample[0])

            if isinstance(module, BasicBlock):
                all_modules.append(module.conv2)
            elif isinstance(module, Bottleneck):
                all_modules.append(module.conv3)

        pinned_out['conv1'] = all_modules
        for name, module in model.named_modules():
            if not isinstance(module, (BasicBlock, Bottleneck)):
               continue

            if module.downsample is not None:
                pinned_out[f'{name}.downsample.0'] = all_modules

            if isinstance(module, BasicBlock):
                pinned_out[f'{name}.conv2'] = all_modules
            else:
                pinned_out[f'{name}.conv3'] = all_modules   

    return pinned_out