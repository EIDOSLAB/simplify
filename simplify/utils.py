#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).

import copy
import re
from typing import Any, Dict, Type, Iterable, Tuple, List

import torch
import torch.nn as nn
from torch import fx
from torchvision.models import MobileNetV3, ShuffleNetV2
from torchvision.models.densenet import _DenseLayer


def matches_module_pattern(pattern: Iterable[Type], node: fx.Node, modules: Dict[str, Any]) -> bool:
    """
    Reimplementation of PyTorch experimental function `matches_module_pattern`.
    See https://github.com/pytorch/pytorch/blob/master/torch/fx/experimental/optimization.py#L26
    """
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


def get_bn_folding(model: nn.Module) -> List[Tuple[str, str]]:
    """
    Search for tuples of adjacent `nn.Conv2d` and `nn.BatchNorm2d` modules.

    Args:
        model (nn.Module): Model on which to perform the conv-bn search.

    Returns:
        List[Tuple[str, str]]: List of the found modules.
    """
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
                    bn_folding.append((node.args[0].target, node.target))

    except Exception as e:
        last_module = None

        for name, module in model.named_modules():
            if isinstance(module, _DenseLayer):
                last_module = None
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                last_module = (name, module)
            if isinstance(module, nn.BatchNorm2d):
                if last_module is not None and last_module[1].weight.shape[0] == module.weight.shape[0]:
                    bn_folding.append((last_module[0], name))

    return bn_folding


def get_previous_layer(connections: Dict, module: fx.Node) -> fx.Node:
    """
    Recursively find the node the precedes `module` in the dictionary of nodes `connections`.

    Args:
        connections (Dict): Dictionary of nodes.
        module (fx.Node): target node.

    Returns:
        fx.Node: Found node.
    """
    for k in connections:
        if any([c == module for c in connections[k]["next"]]):
            if not isinstance(connections[k]["class"], (nn.Conv2d, nn.BatchNorm2d)):
                return get_previous_layer(connections, k)
            else:
                return k


def get_pinned(model: torch.nn.Module) -> List[str]:
    """
    Try to find all the modules for which the output shape needs to stay fixed, (e.g. modules involved in residual connections with a sum).

    Args:
        model (torch.nn.Module): The model on which to perform the research.

    Returns:
        Dict[str]: Dictionary of all the found modules.
    """
    fx_model = fx.symbolic_trace(copy.deepcopy(model))
    modules = dict(fx_model.named_modules())

    names = {re.sub('[_.]', '', n): n for n in modules}

    connections = {}

    # Build dictionary node -> list of connected nodes
    for i, node in enumerate(fx_model.graph.nodes):
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

    # Add input node of CONV with grouping, layer.6 for MNASNet and fc2 for MobileNetV3
    for i, node in enumerate(fx_model.graph.nodes):
        if (isinstance(model, MobileNetV3) and "fc2" in node.name) or \
                (isinstance(model, ShuffleNetV2) and (node.name == "conv1_1" or
                                                      "branch1_3" in node.name or
                                                      "branch2_1" in node.name or
                                                      "branch2_6" in node.name)):
            same_next.add(str(node.name))
        name = node.name.replace("_", ".")
        if re.sub('[_.]', '', name) in names:
            module = modules[names[re.sub('[_.]', '', name)]]
            if isinstance(module, nn.Conv2d) and module.groups > 1:
                same_next.add(str(node.prev))

    # For each node not CONV nor BN recover the closest previous CONV or BN
    to_pin = []
    for m in same_next:
        if not isinstance(connections[m]["class"], (nn.Conv2d, nn.BatchNorm2d)):
            to_pin.append(get_previous_layer(connections, m))
        else:
            to_pin.append(m)

    return [names[re.sub('[_.]', '', n)] for n in list(set(to_pin))]
