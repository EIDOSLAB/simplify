import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch import fx
from torch.fx._experimental.fuser import matches_module_pattern
from torchvision.models import GoogLeNet
from torchvision.models.densenet import _DenseLayer
from torchvision.models.mnasnet import MNASNet, _InvertedResidual as InvertedResidual_MNASNet
from torchvision.models.mobilenetv2 import MobileNetV2, InvertedResidual as InvertedResidual_MobileNetV2
from torchvision.models.mobilenetv3 import MobileNetV3, InvertedResidual as InvertedResidual_MobileNetV3
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
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
    # pinned_out = []
    #
    # if isinstance(model, ResNet):
    #     pinned_out = ['conv1']
    #
    #     last_module = None
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             if module.groups > 1 and last_module is not None:
    #                 pinned_out.append(last_module)
    #             last_module = name
    #
    #         if not isinstance(module, (BasicBlock, Bottleneck)):
    #             continue
    #
    #         block_last = f'{name}.conv2'
    #         if isinstance(module, Bottleneck):
    #             block_last = f'{name}.conv3'
    #         pinned_out.append(block_last)
    #
    #         if module.downsample is not None:
    #             pinned_out.append(f'{name}.downsample.0')
    #
    # elif isinstance(model, MobileNetV3):
    #     pinned_out = ['features.0.0']
    #
    #     last_module, last_block = None, None
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             if module.groups > 1 and last_module is not None:
    #                 pinned_out.append(name)
    #                 pinned_out.append(last_module)
    #             last_module = name
    #
    #         if isinstance(module, InvertedResidual_MobileNetV3):
    #             block_len = len(module.block)
    #             if module.use_res_connect:
    #                 pinned_out.append(f'{name}.block.{block_len - 1}.0')
    #                 if last_block is not None:
    #                     pinned_out.append(f'{last_block[0]}.block.{len(last_block[1].block) - 1}.0')
    #             last_block = (name, module)
    #
    #         if 'fc2' in name:
    #             pinned_out.append(name)
    #
    # elif isinstance(model, MobileNetV2):
    #     pinned_out = ['features.0.0']
    #
    #     last_module, last_block = None, None
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             if module.groups > 1 and last_module is not None:
    #                 pinned_out.append(last_module)
    #             last_module = name
    #
    #         if isinstance(module, InvertedResidual_MobileNetV2):
    #             block_len = len(module.conv)
    #             if module.use_res_connect:
    #                 pinned_out.append(f'{name}.conv.{block_len - 2}')
    #                 if last_block is not None:
    #                     pinned_out.append(f'{last_block[0]}.conv.{len(last_block[1].conv) - 2}')
    #             last_block = (name, module)
    #
    # elif isinstance(model, MNASNet):
    #     pinned_out = ['layers.6']
    #
    #     last_module, last_block = None, None
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             if module.groups > 1 and last_module is not None:
    #                 pinned_out.append(last_module)
    #             last_module = name
    #
    #         if isinstance(module, InvertedResidual_MNASNet):
    #             if module.apply_residual:
    #                 block_len = len(module.layers)
    #                 pinned_out.append(f'{name}.layers.{block_len - 2}')
    #                 if last_block is not None:
    #                     pinned_out.append(f'{last_block[0]}.layers.{len(last_block[1].layers) - 2}')
    #             last_block = (name, module)
    #
    # elif isinstance(model, ShuffleNetV2):
    #     pinned_out = ['conv1.0']
    #
    #     last_module = None
    #     for name, module in model.named_modules():
    #         # print(name)
    #         # works for modules within same branch
    #         if isinstance(module, nn.Conv2d):
    #             if module.groups > 1 and last_module is not None:
    #                 pinned_out.append(last_module)
    #             last_module = name
    #
    #         if isinstance(module, InvertedResidual_ShuffleNetV2):
    #             if len(module.branch1) > 0:
    #                 pinned_out.append(f'{name}.branch1.{len(module.branch1) - 3}')
    #
    #             if len(module.branch2) > 0:
    #                 pinned_out.append(f'{name}.branch2.{len(module.branch2) - 3}')

    pinned_out_graph = []
    if not isinstance(model, GoogLeNet):
        fx_model = fx.symbolic_trace(model)
        modules = dict(fx_model.named_modules())

        last_module = None

        for i, node in enumerate(fx_model.graph.nodes):
            # print(node.all_input_nodes, "->", node.name, "->", list(node.users.keys()))
            if node.target in modules and isinstance(
                    modules[node.target], nn.Conv2d):
                if modules[node.target].groups > 1 and last_module is not None:
                    if last_module.target is not None and last_module.target not in pinned_out_graph:
                        pinned_out_graph.append(last_module.target)
                        # print("Appended", last_module.target)
                last_module = node

            if i > 0 and (len(node.all_input_nodes) >
                          1 or len(node.users) > 1):
                for input_node in node.all_input_nodes:
                    if input_node.target in modules and isinstance(modules[input_node.target],
                                                                   (nn.Conv2d, nn.BatchNorm2d)):
                        if input_node.target is not None and input_node.target not in pinned_out_graph:
                            pinned_out_graph.append(input_node.target)
                            # print("Appended", input_node.target)
                    else:
                        previous_layer = get_previous_layer(
                            input_node, modules)
                        if previous_layer is not None and previous_layer not in pinned_out_graph:
                            pinned_out_graph.append(previous_layer)
                            # print("Appended", previous_layer)

    return pinned_out_graph


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
