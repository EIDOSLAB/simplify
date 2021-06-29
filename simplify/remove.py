from typing import List

import torch
import torch.nn as nn

from .layers import ConvExpand, BatchNormExpand, ConvB


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
        """
        Parameters:
            input: idx of previously remaining channels (0 if channel is pruned, 1 if channel is not pruned)
            output: same for current layer
        """
        input = input[0][0]  # get first item of batch
        
        # Remove input channels
        nonzero_idx = ~(input.view(input.shape[0], -1).sum(dim=1) == 0)
        
        if isinstance(module, nn.Conv2d):
            if module.groups == 1:
                module.weight.data = module.weight.data[:, nonzero_idx]
                module.in_channels = module.weight.shape[1]
            # TODO: handle when groups > 1 (if possible)
        
        elif isinstance(module, nn.Linear):
            module.weight.data = module.weight.data[:, nonzero_idx]
            module.in_features = module.weight.shape[1]
        
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data = module.weight.data[nonzero_idx]
            module.num_features = module.weight.shape[0]
        
        # Compute remaining channels indices
        output = torch.ones_like(output) * float('nan')
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            return output
        
        # If not pinned: remove zeroed output channels
        if not isinstance(module, nn.BatchNorm2d):
            shape = module.weight.shape

            # If module is ConvExpand, expand weights back to original
            # adding zero where channels were pruned
            # so that new idxs are updated accordingly
            if isinstance(module, ConvExpand):
                zeros = torch.zeros(1, *shape[1:])
                expanded_weight = torch.cat((module.weight.data, zeros), dim=0)
                expanded_weight = expanded_weight[module.idxs]
                nonzero_idx = ~(expanded_weight.view(expanded_weight.shape[0], -1).sum(dim=1) == 0)
                module.weight.data = expanded_weight.data[nonzero_idx]
                
            else:
                nonzero_idx = ~(module.weight.view(shape[0], -1).sum(dim=1) == 0)
                module.weight.data = module.weight.data[nonzero_idx]
        
        if name in pinned_out:
            idxs = []
            current = 0
            zero_idx = torch.where(~nonzero_idx)[0]
            for i in range(module.weight.data.shape[0] + len(zero_idx)):
                if i in zero_idx:
                    idxs.append(module.weight.data.shape[0])
                else:
                    idxs.append(current)
                    current += 1
            if isinstance(module, nn.Conv2d):
                #print(f'Transforming {name} into ConvExpand')
                module = ConvExpand.from_conv(module, idxs, module.bf)

                #########################################################
                ## Given that ConvExpand were expanded back to Conv2d, ##
                ## this is no longer neeeded                           ##
                #########################################################

                #if isinstance(module, ConvB):
                #    print(f'Transforming {name} into ConvExpand')
                #    module = ConvExpand.from_conv(module, idxs, module.bf)
                #else:
                #    if getattr(module, 'bias', None) is not None:
                #        module.bias.data = module.bias.data[nonzero_idx]
                    # TODO: maybe this is too much, if bf is 0 there is no need for the addition
                #    print(f'Transforming {name} into ConvExpand (prev idxs: {len(module.idxs)}, new idxs: {len(idxs)})')
                #    module = ConvExpand.from_conv(module, idxs, torch.zeros_like(output[0]))
                
            if isinstance(module, nn.BatchNorm2d):
                bf = module.bias.data.mul(~nonzero_idx)
                module.running_mean.data = module.running_mean.data[nonzero_idx]
                module.running_var.data = module.running_var.data[nonzero_idx]
                module.bias.data = module.bias.data[nonzero_idx]
                module = BatchNormExpand.from_bn(module, idxs, bf, output.shape)
        else:
            if getattr(module, 'bias', None) is not None:
                module.bias.data = module.bias.data[nonzero_idx]
            
            if getattr(module, 'bf', None) is not None:
                module.bf.data = module.bf.data[nonzero_idx]
            
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.data = module.running_mean.data[nonzero_idx]
                module.running_var.data = module.running_var.data[nonzero_idx]
            
            output = torch.zeros_like(output)
            output[:, nonzero_idx] = float('nan')
        
        if isinstance(module, nn.Conv2d):
            module.out_channels = module.weight.shape[0]
        elif isinstance(module, nn.Linear):
            module.out_features = module.weight.shape[0]
        
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
