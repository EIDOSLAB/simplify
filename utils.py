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

        for name, module in model.named_modules():
            if not isinstance(module, (BasicBlock, Bottleneck)):
                continue

            block_last = (f'{name}.conv2', module.conv2)
            if isinstance(module, Bottleneck):
                block_last = (f'{name}.conv3', module.conv3)
            
            if module.downsample is not None:
                pinned_out[block_last[0]] = [module.downsample[0]]
                pinned_out[f'{name}.downsample.0'] = [block_last[1]]
                last_module = [block_last, (f'{name}.downsample.0', module.downsample[0])]

            else:
                pinned_out[block_last[0]] = [last[1] for last in last_module]
                for (last_name, _) in last_module:
                    pinned_out[last_name].append(block_last[1])
                last_module = [block_last]

    return pinned_out