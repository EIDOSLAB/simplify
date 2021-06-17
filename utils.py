import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import fx
from torch.fx._experimental.fuser import matches_module_pattern


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
    for input_node in node.all_input_nodes:
        if input_node.target in modules and isinstance(modules[input_node.target], (nn.Conv2d, nn.BatchNorm2d)):
            return input_node.target
        else:
            return get_previous_layer(input_node, modules)


def get_pinned_out(model):
    pinned_out = []
    
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    
    for node in reversed(fx_model.graph.nodes):
        if len(node.all_input_nodes) > 1:
            for input_node in node.all_input_nodes:
                if input_node.target in modules and isinstance(modules[input_node.target], (nn.Conv2d, nn.BatchNorm2d)):
                    if input_node.target not in pinned_out:
                        pinned_out.append(input_node.target)
                else:
                    previous_layer = get_previous_layer(input_node, modules)
                    if previous_layer not in pinned_out:
                        pinned_out.append(previous_layer)
    
    return pinned_out


def get_bn_folding(model):
    bn_folding = []
    patterns = [(torch.nn.Conv2d, torch.nn.BatchNorm2d)]
    fx_model = fx.symbolic_trace(model)
    modules = dict(fx_model.named_modules())
    
    for pattern in patterns:
        for node in fx_model.graph.nodes:
            if matches_module_pattern(pattern, node, modules):
                if len(node.args[0].users) > 1:  # Output of conv is used by other nodes
                    continue
                bn_folding.append([node.args[0].target, node.target])
    
    return bn_folding
