from typing import List

import torch
import torch.nn as nn

from .layers import ConvExpand, BatchNormExpand


@torch.no_grad()
def remove_zeroed(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:
    @torch.no_grad()
    def __remove_nan(module, input):
        nan_idx = torch.isnan(input[0])
        new_input = input[0].clone()
        new_input[~nan_idx] = 0
        new_input[nan_idx] = 1
        return (new_input, *input[1:])
    
    @torch.no_grad()
    def __remove_zeroed_channels_hook(module, input, output, name):
        #print('\n', name, module)

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
        #print('input:', input.shape)

        if isinstance(module, nn.Conv2d):
            if module.groups == 1:
                module.weight = torch.nn.parameter.Parameter(module.weight[:, nonzero_idx])
                module.in_channels = module.weight.shape[1]
            # TODO: handle when groups > 1 (if possible)
        
        elif isinstance(module, nn.Linear):
            module.weight = torch.nn.parameter.Parameter(module.weight[:, nonzero_idx])
            module.in_features = module.weight.shape[1]
        
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data[~nonzero_idx].mul_(0) #= torch.nn.parameter.Parameter(module.weight[nonzero_idx])
            #module.bf.data[~nonzero_idx].mul_(0) #= torch.nn.parameter.Parameter(module.bf[nonzero_idx])
            module.running_mean.data.mul_(~nonzero_idx)
            module.running_var.data[~nonzero_idx] = 1.
            module.num_features = nonzero_idx.sum()
        
        ###########################################
        ##### STEP 2 - REMOVE OUTPUT CHANNELS #####
        ###########################################

        # By default, all channels are remaining
        output = torch.ones_like(output) * float('nan')
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            return output

        shape = module.weight.shape
        
        # 1. Compute remaining channels indices
        nonzero_idx = ~(module.weight.view(shape[0], -1).sum(dim=1) == 0)

        # If module is ConvExpand or BatchNormExpand, expand weights back to original
        # adding zero where channels were pruned
        # so that new idxs are updated accordingly
        if isinstance(module, ConvExpand):
            zeros = torch.zeros(1, *shape[1:])
            expanded_weight = torch.cat((module.weight, zeros), dim=0)
            expanded_weight = expanded_weight[module.idxs]
            nonzero_idx = ~(expanded_weight.view(expanded_weight.shape[0], -1).sum(dim=1) == 0)
            module.weight = torch.nn.parameter.Parameter(expanded_weight)
        
        elif isinstance(module, BatchNormExpand):
            # Expand weight
            zeros = torch.zeros(1, *shape[1:])
            expanded_weight = torch.cat((module.weight, zeros), dim=0)
            expanded_weight = expanded_weight[module.idxs]
            nonzero_idx = ~(expanded_weight.view(expanded_weight.shape[0], -1).sum(dim=1) == 0)
            module.weight = torch.nn.parameter.Parameter(expanded_weight)

            # Expand running_mean
            zeros = torch.zeros(1, *module.running_mean.shape[1:])
            expanded_mean = torch.cat((module.running_mean, zeros), dim=0)
            module.running_mean = expanded_mean[module.idxs]

            # Expand running_var
            ones = torch.ones(1, *module.running_var.shape[1:])
            expanded_var = torch.cat((module.running_var, ones), dim=0)
            module.running_var = expanded_var[module.idxs]

        # 2. Remove weight channels
        module.weight = torch.nn.parameter.Parameter(module.weight[nonzero_idx])
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean = module.running_mean[nonzero_idx]
            module.running_var = module.running_var[nonzero_idx]

        # 3. If it is a pinned layer, convert it into ConvExpand or BatchNormExpand
        if name in pinned_out:
            # Compute expansion indices
            idxs = []
            current = 0
            zero_idx = torch.where(~nonzero_idx)[0]
            for i in range(module.weight.data.shape[0] + len(zero_idx)):
                if i in zero_idx:
                    idxs.append(module.weight.data.shape[0])
                else:
                    idxs.append(current)
                    current += 1

            # Keep bias (bf) full size
            if isinstance(module, nn.Conv2d):
                module = ConvExpand.from_conv(module, idxs, module.bf)
                
            if isinstance(module, nn.BatchNorm2d):
                module = BatchNormExpand.from_bn(module, idxs, module.bf, output.shape)
        else:
            if getattr(module, 'bias', None) is not None:
                module.bias = torch.nn.parameter.Parameter(module.bias[nonzero_idx])
            
            if getattr(module, 'bf', None) is not None:
                module.bf = torch.nn.parameter.Parameter(module.bf[nonzero_idx])
            
            output = torch.zeros_like(output)
            output[:, nonzero_idx] = float('nan')
        
        if isinstance(module, nn.Conv2d):
            module.out_channels = module.weight.shape[0]
        elif isinstance(module, nn.Linear):
            module.out_features = module.weight.shape[0]
        elif isinstance(module, nn.BatchNorm2d):
            module.num_features = module.weight.shape[0]
        
        return output
    
    handles = []
    for name, module in model.named_modules():
        # TODO: add all parameters layers
        if not isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            continue
        
        handle = module.register_forward_pre_hook(__remove_nan)
        handles.append(handle)
        handle = module.register_forward_hook(lambda m, i, o, n=name: __remove_zeroed_channels_hook(m, i, o, n))
        handles.append(handle)
    
    x = torch.ones_like(x) * float("nan")
    model(x)
    
    for h in handles:
        h.remove()
    
    return model
