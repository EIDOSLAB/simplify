import os
import random

import numpy as np
from numpy.lib import isin
import torch
from torchvision.models.mobilenetv3 import MobileNetV3
from torchvision.models.mobilenetv3 import InvertedResidual as InvertedResidual_MobileNetV3
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
        pinned_out = ['conv1']
        
        for name, module in model.named_modules():
            if not isinstance(module, (BasicBlock, Bottleneck)):
                continue
            
            block_last = f'{name}.conv2'
            if isinstance(module, Bottleneck):
                block_last = f'{name}.conv3'
            pinned_out.append(block_last)

            if module.downsample is not None:
                pinned_out.append(f'{name}.downsample.0')

    if isinstance(model, MobileNetV3):
        pinned_out = ['features.0.0']

        for name, module in model.named_modules():
            if isinstance(module, InvertedResidual_MobileNetV3):
                block_len = len(module.block)
                pinned_out.append(f'{name}.block.{block_len-1}.0')
    
    print('Pinned layers:', pinned_out)

    return pinned_out
