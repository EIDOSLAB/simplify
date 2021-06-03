import os
import random

import numpy as np
from numpy.lib import isin
from numpy.lib.function_base import insert
import torch
import torch.nn as nn

from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidual as InvertedResidual_MobileNetV3 
from torchvision.models.mobilenetv2 import MobileNetV2, InvertedResidual as InvertedResidual_MobileNetV2
from torchvision.models.mnasnet import MNASNet, _InvertedResidual as InvertedResidual_MNASNet
from torchvision.models.shufflenetv2 import ShuffleNetV2, InvertedResidual as InvertedResidual_ShuffleNetV2

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
        pinned_out = ['conv1']
        
        last_module = None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups > 1 and last_module is not None:
                    pinned_out.append(last_module)
                last_module = name

            if not isinstance(module, (BasicBlock, Bottleneck)):
                continue
            
            block_last = f'{name}.conv2'
            if isinstance(module, Bottleneck):
                block_last = f'{name}.conv3'
            pinned_out.append(block_last)

            if module.downsample is not None:
                pinned_out.append(f'{name}.downsample.0')

    elif isinstance(model, MobileNetV3):
        pinned_out = ['features.0.0']

        last_module, last_block = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups > 1 and last_module is not None:
                    pinned_out.append(name)
                    pinned_out.append(last_module)
                last_module = name

            if isinstance(module, InvertedResidual_MobileNetV3):
                block_len = len(module.block)
                if module.use_res_connect:
                    pinned_out.append(f'{name}.block.{block_len-1}.0')
                    if last_block is not None:
                        pinned_out.append(f'{last_block[0]}.block.{len(last_block[1].block)-1}.0')
                last_block = (name, module)

            if 'fc2' in name:
                pinned_out.append(name)

    elif isinstance(model, MobileNetV2):
        pinned_out = ['features.0.0']

        last_module, last_block = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups > 1 and last_module is not None:
                    pinned_out.append(last_module)
                last_module = name

            if isinstance(module, InvertedResidual_MobileNetV2):
                block_len = len(module.conv)
                if module.use_res_connect:
                    pinned_out.append(f'{name}.conv.{block_len-2}')
                    if last_block is not None:
                        pinned_out.append(f'{last_block[0]}.conv.{len(last_block[1].conv)-2}')
                last_block = (name, module)

    elif isinstance(model, MNASNet):
        pinned_out = ['layers.6']

        last_module, last_block = None, None
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if module.groups > 1 and last_module is not None:
                    pinned_out.append(last_module)
                last_module = name

            if isinstance(module, InvertedResidual_MNASNet):
                if module.apply_residual:
                    block_len = len(module.layers)
                    pinned_out.append(f'{name}.layers.{block_len-2}')
                    if last_block is not None:
                        pinned_out.append(f'{last_block[0]}.layers.{len(last_block[1].layers)-2}')
                last_block = (name, module)

    elif isinstance(model, ShuffleNetV2):
        pinned_out = ['conv1.0']

        last_module = None
        for name, module in model.named_modules():
            # works for modules within same branch
            if isinstance(module, nn.Conv2d):
                if module.groups > 1 and last_module is not None:
                    pinned_out.append(last_module)
                last_module = name

            if isinstance(module, InvertedResidual_ShuffleNetV2):
                if len(module.branch1) > 0:
                    pinned_out.append(f'{name}.branch1.{len(module.branch1)-3}')
            
                if len(module.branch2) > 0:
                    pinned_out.append(f'{name}.branch2.{len(module.branch2)-3}')
                   
    return pinned_out

