from os import error
from typing import List

import torch
import torch.nn as nn

from .layers import BatchNormB, BatchNormExpand, ConvB, ConvExpand


@torch.no_grad()
def propagate_bias(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:
    @torch.no_grad()
    def __remove_nan(module, input):
        module.register_buffer("pruned_input", ~torch.isnan(input[0][0].view(input[0][0].shape[0], -1).sum(dim=1)))
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
        
        ###########################################################################################
        ## STEP 1. Fuse biases of pruned channels in the previous module into the current module ##
        ###########################################################################################

        input = input[0]
        bias_feature_maps = output[0].clone()
        
        if isinstance(module, nn.Conv2d):
            # For a conv layer, we remove the scalar biases
            # and use bias matrices (ConvB)
            # if bias_feature_maps.abs().sum() != 0.:
            # remove native bias
            if getattr(module, 'bias', None) is not None:
                module.register_parameter('bias', None)

            if not isinstance(module, ConvExpand):
                module = ConvB.from_conv(module, bias_feature_maps)
            else: # if it is already ConvExpand, just update bf
                module.register_parameter('bf', nn.Parameter(bias_feature_maps))

        elif isinstance(module, nn.BatchNorm2d):
            bias_feature_maps = bias_feature_maps[:, 0, 0].mul(module.pruned_input)
            
            if getattr(module, 'bias', None) is not None:
                bias_feature_maps += module.bias.data.mul(~module.pruned_input)
                module.register_parameter('bias', None)
            elif getattr(module, 'bf', None) is not None:
                bias_feature_maps += module.bf.mul(~module.pruned_input)

            if not isinstance(module, BatchNormExpand):
                module = BatchNormB.from_bn(module, bias_feature_maps)
            else:  # if it is already BatchNormExpand, just update bf
                module.register_parameter('bf', nn.Parameter(bias_feature_maps))
            
            # Mark corresponding weights to be pruned
            module.weight.data.mul_(~module.pruned_input)  
        
        elif isinstance(module, nn.Linear):
            # TODO: handle missing bias
            # For a linear layer, we can just update the scalar bias values
            #if getattr(module, 'bias', None) is not None:
            #    module.bias.data = bias_feature_maps
            module.register_parameter('bias', nn.Parameter(bias_feature_maps))

        else:
            error('Unsupported module type:', module)
        
        #############################################
        ## STEP 2. Propagate output to next module ##
        #############################################

        shape = module.weight.shape  # Compute mask of zeroed (pruned) channels
        pruned_channels = module.weight.view(shape[0], -1).sum(dim=1) == 0
        
        if name in pinned_out or (isinstance(module, nn.Conv2d) and module.groups > 1):
            # No bias is propagated for pinned layers
            return output * float('nan')
        
        # Propagate the pruned channels and the corresponding bias if present
        # Zero out biases of pruned channels in current layer
        if isinstance(module, nn.Linear):
            output[~pruned_channels[None, :].expand_as(output)] *= float('nan')
            if getattr(module, 'bias', None) is not None:
                module.bias.data.mul_(~pruned_channels)
        
        elif isinstance(module, nn.Conv2d):
            output[~pruned_channels[None, :, None, None].expand_as(output)] *= float('nan')
            if isinstance(module, (ConvB, ConvExpand)):
                if getattr(module, 'bf', None) is not None:
                    module.bf.data.mul_(~pruned_channels[:, None, None])
            else:
                if getattr(module, 'bias', None) is not None:
                    module.bias.data.mul_(~pruned_channels)
        
        if isinstance(module, nn.BatchNorm2d):
            output[~pruned_channels[None, :, None, None].expand_as(output)] *= float('nan')
            if isinstance(module, (BatchNormB, BatchNormExpand)):
                module.bf.data.mul_(~pruned_channels)
            else:
                if getattr(module, 'bias', None) is not None:
                    module.bias.data.mul_(~pruned_channels)
            module.running_mean.data.mul_(~pruned_channels)
            module.running_var.data[pruned_channels] = 1.

        del module._buffers["pruned_input"]
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
