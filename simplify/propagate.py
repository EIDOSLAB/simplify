import torch
import torch.nn as nn

from os import error
from typing import List
from .layers import ConvB

@torch.no_grad()
def propagate_bias(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:

    @torch.no_grad()
    def __remove_nan(module, input):
        if torch.isnan(input[0]).sum() > 0:
            input[0][torch.isnan(input[0])] = 0
        return input
    
    @torch.no_grad()
    def __propagate_biases_hook(module, input, output, name=None):
        """
        Parameters:
            - module: nn.module
            - input: torch.Tensor; non-zero channels correspond to remaining biases pruned channels in the previos module,
                    and thus should be used to update the current biases
            - output: torch.Tensor
        """

        # STEP 1. Fuse biases of pruned channels in the previous module into the current module
        input = input[0]
        bias_feature_maps = output[0].clone()
        
        if isinstance(module, nn.Conv2d):
            # For a conv layer, we remove the scalar biases
            # and use bias matrices (ConvB)
            if getattr(module, 'bias', None) is not None:
                module.register_parameter('bias', None)
            module = ConvB.from_conv(module, bias_feature_maps)
        
        elif isinstance(module, nn.Linear):
            # TODO: if bias is missing, it must be inserted here
            # For a linear layer, we can just update the scalar bias values
            if getattr(module, 'bias', None) is not None:
                module.bias.copy_(bias_feature_maps)
        
        elif isinstance(module, nn.BatchNorm2d):
            pruned_input = input.squeeze(dim=0)
            pruned_input = pruned_input.view(pruned_input.shape[0], -1).sum(dim=1) != 0

            # TODO: if bias is missing, it must be inserted here
            if getattr(module, 'bias', None) is not None:
                module.bias[pruned_input].copy_(bias_feature_maps[:, 0, 0][pruned_input])
            module.weight.data.mul_(~pruned_input) # Mark corresponding weights to be pruned

        else:
            error('Unsupported module type:', module)
        
        # STEP 2. Propagate output to next module
        shape = module.weight.shape  # Compute mask of zeroed (pruned) channels
        pruned_channels = module.weight.view(shape[0], -1).sum(dim=1) == 0
        
        if name in pinned_out:
            # No bias is propagated for pinned layers
            return output * float('nan')
        
        if getattr(module, 'bias', None) is not None or getattr(module, 'bf', None) is not None:
            # Propagate only the bias values corresponding to pruned channels
            # Zero out biases of pruned channels in current layer
            if isinstance(module, nn.Linear):
                output[~pruned_channels[None, :].expand_as(output)] *= float('nan')
                module.bias.data.mul_(~pruned_channels)
            
            elif isinstance(module, ConvB):
                output[~pruned_channels[None, :, None, None].expand_as(output)] *= float('nan')
                module.bf.data.mul_(~pruned_channels[:, None, None])
                
            if isinstance(module, nn.BatchNorm2d):
                output[~pruned_channels[None, :, None, None].expand_as(output)] *= float('nan')
                module.bias.data.mul_(~pruned_channels)
                module.running_mean.data.mul_(~pruned_channels)
                module.running_var.data[pruned_channels] = 1.
        
        return output
    
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            handle = module.register_forward_pre_hook(__remove_nan)
            handles.append(handle)
            handle = module.register_forward_hook(lambda m, i, o, n=name: __propagate_biases_hook(m, i, o, n))
            handles.append(handle)
    
    # Propagate biases
    zeros = torch.zeros_like(x)  # make sure input is zero
    model(zeros)
    
    for h in handles:
        h.remove()
    
    return model