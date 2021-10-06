import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import fx
from torchvision.models.densenet import _DenseLayer


def matches_module_pattern(pattern, node, modules):
    if len(node.args) == 0:
        return False
    nodes = (node.args[0], node)
    for expected_type, current_node in zip(pattern, nodes):
        if not isinstance(current_node, fx.Node):
            return False
        if current_node.op != 'call_module':
            return False
        if not isinstance(current_node.target, str):
            return False
        if current_node.target not in modules:
            return False
        if type(modules[current_node.target]) is not expected_type:
            return False
    return True


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
        if input_node.target in modules and isinstance(modules[input_node.target], (nn.Conv2d, nn.BatchNorm2d)):
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
            print(node.name)
            if node.target in modules and isinstance(modules[node.target], nn.Conv2d):
                if modules[node.target].groups > 1 and last_module is not None:
                    if last_module.target is not None and last_module.target not in pinned_out:
                        pinned_out.append(last_module.target)
                last_module = node
            
            if i > 0 and (len(node.all_input_nodes) > 1 or len(node.users) > 1):
                for input_node in node.all_input_nodes:
                    if input_node.target in modules and isinstance(modules[input_node.target],
                                                                   (nn.Conv2d, nn.BatchNorm2d)):
                        if input_node.target is not None and input_node.target not in pinned_out:
                            pinned_out.append(input_node.target)
                    else:
                        previous_layer = get_previous_layer(input_node, modules)
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


def get_previous_layer_2(connections, module):
    for k in connections:
        if any([c == module for c in connections[k]["next"]]):
            if not isinstance(connections[k]["class"], (nn.Conv2d, nn.BatchNorm2d)):
                return get_previous_layer(connections, k)
            else:
                return k


def get_pinned(model):
    fx_model = fx.symbolic_trace(copy.deepcopy(model))
    modules = dict(fx_model.named_modules())
    
    connections = {}
    
    # Build dictionary node -> list of connected nodes
    for i, node in enumerate(fx_model.graph.nodes):
        # print(f"{node.name}->{[str(user) for user in node.users]}")
        if node.target in modules:
            module = modules[node.target]
        else:
            module = None
        
        connections[node.name] = {"next": [str(user) for user in node.users], "class": module}
    
    # Remove duplicates and build list of "to-pin" nodes (may contain nodes not CONV nor BN)
    same_next = []
    for k in connections:
        for k2 in connections:
            if k != k2:
                if "add" in str(set(connections[k]["next"]) & set(connections[k2]["next"])):
                    same_next.append([k, k2])
    
    same_next = set([item for sublist in same_next for item in sublist])
    
    # For each node not CONV nor BN recover the closest previous CONV or BN
    to_pin = []
    for m in same_next:
        if not isinstance(connections[m]["class"], (nn.Conv2d, nn.BatchNorm2d)):
            to_pin.append(get_previous_layer_2(connections, m))
        else:
            to_pin.append(m)
    
    return list(set(to_pin))
