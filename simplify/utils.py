import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import fx
# from torch.fx.experimental.optimization import matches_module_pattern
from torch.fx._experimental.fuser import matches_module_pattern

from torchvision.models.densenet import _DenseLayer


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


def get_previous_layer(node, modules):
    # print("get_previous_layer")
    for input_node in node.all_input_nodes:
        # print(input_node.name)
        if input_node.target in modules and isinstance(
                modules[input_node.target], (nn.Conv2d, nn.BatchNorm2d)):
            return input_node.target
        else:
            return get_previous_layer(input_node, modules)


def get_pinned_out(model):
    pinned_out = []
    try:
        fx_model = fx.symbolic_trace(copy.deepcopy(model))
        modules = dict(fx_model.named_modules())
        
        last_module = None
        
        for i, node in enumerate(fx_model.graph.nodes):
            if node.target in modules and isinstance(
                    modules[node.target], nn.Conv2d):
                if modules[node.target].groups > 1 and last_module is not None:
                    if last_module.target is not None and last_module.target not in pinned_out:
                        pinned_out.append(last_module.target)
                last_module = node
            
            if i > 0 and (len(node.all_input_nodes) >
                          1 or len(node.users) > 1):
                for input_node in node.all_input_nodes:
                    if input_node.target in modules and isinstance(modules[input_node.target],
                                                                   (nn.Conv2d, nn.BatchNorm2d)):
                        if input_node.target is not None and input_node.target not in pinned_out:
                            pinned_out.append(input_node.target)
                    else:
                        previous_layer = get_previous_layer(
                            input_node, modules)
                        if previous_layer is not None and previous_layer not in pinned_out:
                            pinned_out.append(previous_layer)
    except Exception as e:
        pass
    
    return pinned_out


def get_bn_folding(model):
    bn_folding = []
    
    try:
        patterns = [(torch.nn.Conv2d, torch.nn.BatchNorm2d)]
        fx_model = fx.symbolic_trace(model)
        modules = dict(fx_model.named_modules())
        
        for pattern in patterns:
            for node in fx_model.graph.nodes:
                if matches_module_pattern(pattern, node, modules):
                    if len(node.args[0].users) > 1:
                        continue
                    bn_folding.append([node.args[0].target, node.target])
    
    except Exception as e:
        last_module = None
        
        for name, module in model.named_modules():
            if isinstance(module, _DenseLayer):
                last_module = None
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                last_module = (name, module)
            if isinstance(module, nn.BatchNorm2d):
                if last_module is not None and last_module[1].weight.shape[0] == module.weight.shape[0]:
                    bn_folding.append([last_module[0], name])
    
    return bn_folding
