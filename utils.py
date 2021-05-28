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
        pinned_out = {}
        last_module = model.conv1

        for name, module in model.named_modules():
            if isinstance(module, BasicBlock):
                if module.downsample is not None:
                    pinned_out[f'{name}.conv2'] = module.downsample[0]
                    pinned_out[f'{name}.downsample.0'] = module.conv2
                    last_module = [module.conv2, module.downsample[0]]
                else:
                    pinned_out[f'{name}.conv2'] = last_module
                    last_module = [module.conv2]
            
            if isinstance(module, Bottleneck):
                if module.downsample is not None:
                    pinned_out[f'{name}.conv3'] = module.downsample[0]
                    pinned_out[f'{name}.downsample.0'] = module.conv3
                    last_module = [module.conv3, module.downsample[0]]
                else:
                    pinned_out[f'{name}.conv3'] = last_module
                    last_module = [module.conv3]

    return pinned_out