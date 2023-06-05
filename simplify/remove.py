#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).

from typing import List

import torch
import torch.nn as nn

from .layers import BatchNormB, ConvExpand, BatchNormExpand, LinearExpand, ConvB


@torch.no_grad()
def remove_zeroed(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:
    @torch.no_grad()
    def __remove_nan(module, input, name):
        nan_idx = torch.isnan(input[0])
        new_input = input[0].clone()
        new_input[~nan_idx] = 0
        new_input[nan_idx] = 1
        return (new_input, *input[1:])

    @torch.no_grad()
    def __remove_zeroed_channels_hook(module, input, output, name):
        # print('\n', name, module)
        """
        Parameters:
            input: idx of previously remaining channels (0 if channel is pruned, 1 if channel is not pruned)
            output: same for current layer
        """
        input = input[0][0]  # get first item of batch

        ##########################################
        ##### STEP 1 - REMOVE INPUT CHANNELS #####
        ##########################################

        # Compute non-zero input channels indices
        nonzero_idx = ~(input.view(input.shape[0], -1).sum(dim=1) == 0)
        # print('input:', input.shape)

        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(module.weight[:, nonzero_idx])
            module.in_features = module.weight.shape[1]

        elif isinstance(module, nn.Conv2d):
            if module.groups == 1:
                module.weight = nn.Parameter(module.weight[:, nonzero_idx])
                module.in_channels = module.weight.shape[1]
            # TODO: handle when groups > 1 (if possible)

        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.mul_(nonzero_idx)
            module.running_mean.data.mul_(nonzero_idx)
            module.running_var.data[~nonzero_idx] = 1.
            module.num_features = nonzero_idx.sum()

        ###########################################
        ##### STEP 2 - REMOVE OUTPUT CHANNELS #####
        ###########################################

        # By default, all channels are remaining
        output = torch.ones_like(output) * float('nan')
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            return output

        # if training and name in pinned_out:
        #     return output

        shape = module.weight.shape

        # 1. Compute remaining channels indices
        nonzero_idx = ~(torch.abs(module.weight.view(shape[0], -1)).sum(dim=1) == 0)

        # If module is ConvExpand or BatchNormExpand, expand weights back to original
        # adding zero where channels were pruned so that new idxs are updated accordingly
        if isinstance(module, ConvExpand):
            zeros = torch.zeros(module.bf.shape[0], *module.weight.shape[1:], device=module.weight.device)
            index = module.idxs[:, None, None, None].expand_as(module.weight)
            expanded_weight = torch.scatter(zeros, 0, index, module.weight)

            nonzero_idx = ~(expanded_weight.view(expanded_weight.shape[0], -1).sum(dim=1) == 0)
            module.weight = nn.Parameter(expanded_weight)

        elif isinstance(module, BatchNormExpand):
            # Expand weight
            zeros = torch.zeros(1, *shape[1:], device=module.weight.device)
            expanded_weight = torch.cat((module.weight, zeros), dim=0)
            expanded_weight = expanded_weight[module.expansion_idxs]
            nonzero_idx = ~(expanded_weight.view(expanded_weight.shape[0], -1).sum(dim=1) == 0)
            module.weight = nn.Parameter(expanded_weight)

            # Expand running_mean
            zeros = torch.zeros(1, *module.running_mean.shape[1:], device=module.weight.device)
            expanded_mean = torch.cat((module.running_mean, zeros), dim=0)
            module.running_mean = expanded_mean[module.expansion_idxs]

            # Expand running_var
            ones = torch.ones(1, *module.running_var.shape[1:], device=module.weight.device)
            expanded_var = torch.cat((module.running_var, ones), dim=0)
            module.running_var = expanded_var[module.expansion_idxs]

        # 2. Remove weight channels
        module.weight = nn.Parameter(module.weight[nonzero_idx])
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean = module.running_mean[nonzero_idx]
            module.running_var = module.running_var[nonzero_idx]

        # 3. If it is a pinned layer, convert it into LinearExpand, ConvExpand or BatchNormExpand
        if name in pinned_out:
            idxs = torch.where(nonzero_idx)[0]

            if isinstance(module, nn.Linear):
                module = LinearExpand.from_linear(module, idxs, module.bias)

            # Keep bias (bf) full size
            elif isinstance(module, nn.Conv2d):
                bias = module.bf if isinstance(module, ConvB) else module.bias
                module = ConvExpand.from_conv(module, idxs, bias, isinstance(module, ConvB), output.shape)

            elif isinstance(module, nn.BatchNorm2d):
                bias = module.bf if isinstance(module, BatchNormB) else module.bias
                module = BatchNormExpand.from_bn(module, idxs, bias, output.shape)

            if not isinstance(module, (ConvB, BatchNormB)):
                module.register_parameter("bias", None)
        else:
            if getattr(module, 'bf', None) is not None:
                module.bf = nn.Parameter(module.bf[nonzero_idx])

            output = torch.zeros_like(output)
            output[:, nonzero_idx] = float('nan')

        if getattr(module, 'bias', None) is not None:
            module.bias = nn.Parameter(module.bias[nonzero_idx])

        if isinstance(module, nn.Conv2d):
            module.out_channels = module.weight.shape[0]
        elif isinstance(module, nn.Linear):
            module.out_features = module.weight.shape[0]
        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = module.weight.shape[0]

        # print(f"layer shape {module.weight.shape}")
        return output

    handles = []
    for name, module in model.named_modules():
        # TODO: add all parameters layers
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            continue

        handle = module.register_forward_pre_hook(lambda m, i, n=name: __remove_nan(m, i, n))
        handles.append(handle)
        handle = module.register_forward_hook(lambda m, i, o, n=name: __remove_zeroed_channels_hook(m, i, o, n))
        handles.append(handle)

    x = torch.ones_like(x) * float("nan")
    model(x)

    for h in handles:
        h.remove()

    return model
