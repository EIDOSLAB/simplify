#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).

from os import error
from typing import List

import torch
import torch.nn as nn

from .layers import BatchNormB, BatchNormExpand, ConvB, ConvExpand


@torch.no_grad()
def propagate_bias(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:
    """
    Propagate a pruned neuron non-zero bias to the next layers non-pruned neurons.

    Args:
        model (nn.Module):
        x (torch.Tensor): `model`'s input of shape [1, C, N, M], same as the model usual input.
        pinned_out (List): List of `nn.Modules` which output needs to remain of the original shape (e.g. layers related to a residual connection with a sum operation).

    Returns:
        nn.Module: Model with propagated bias.

    """
    
    @torch.no_grad()
    def __remove_nan(module, input):
        """
        PyTorch hook that removes nans from input.
        """
        module.register_buffer("pruned_input", ~torch.isnan(input[0][0].view(input[0][0].shape[0], -1).sum(dim=1)))
        if torch.isnan(input[0]).sum() > 0:
            input[0][torch.isnan(input[0])] = 0
        return input
    
    @torch.no_grad()
    def __propagate_biases_hook(module, input, output):
        """
        PyTorch hook used to propagate the biases of pruned neurons to following non-pruned layers.
        """
        
        ###########################################################################################
        ## STEP 1. Fuse biases of pruned channels in the previous module into the current module ##
        ###########################################################################################
        
        bias_feature_maps = output[0].clone()
        
        if isinstance(module, nn.Conv2d):
            # For a conv layer, we remove the scalar biases
            # and use bias matrices (ConvB)
            if bias_feature_maps.abs().sum() != 0.:
                # remove native bias
                if getattr(module, 'bias', None) is not None:
                    module.register_parameter('bias', None)
                
                all_unique = True
                
                for i in range(bias_feature_maps.shape[0]):
                    uniq = torch.unique(bias_feature_maps[i])
                    if uniq.shape[0] > 1:
                        all_unique = False
                        break
                
                if not all_unique:
                    if not isinstance(module, ConvExpand):
                        module = ConvB.from_conv(module, bias_feature_maps)
                    else:  # if it is already ConvExpand, just update bf
                        module.register_parameter('bf', nn.Parameter(bias_feature_maps))
                else:
                    module.register_parameter("bias", nn.Parameter(bias_feature_maps[:, 0, 0]))
        
        elif isinstance(module, nn.BatchNorm2d):
            pruned_input = module.pruned_input
            if isinstance(module, BatchNormExpand):
                ones = torch.ones(1, device=module.weight.device).bool()
                expanded_pruned_input = torch.cat((module.pruned_input, ones), dim=0)
                expanded_pruned_input = expanded_pruned_input[module.idxs]
                module.pruned_input = expanded_pruned_input
            
            # if module.pruned_input.shape[0] != bias_feature_maps.shape[0]:
            #    print(module)
            #    print(bias_feature_maps.shape, module.pruned_input.shape)
            bias_feature_maps = bias_feature_maps[:, 0, 0].mul(module.pruned_input)
            
            if getattr(module, 'bias', None) is not None:
                bias_feature_maps += module.bias.data.mul(~module.pruned_input)
                module.register_parameter('bias', None)
            
            elif getattr(module, 'bf', None) is not None:
                bias_feature_maps += module.bf.mul(~module.pruned_input)
            
            # Restore compressed indices
            module.pruned_input = pruned_input
            
            if not isinstance(module, BatchNormExpand):
                # module = BatchNormB.from_bn(module, bias_feature_maps)
                module.bias = torch.nn.parameter.Parameter(bias_feature_maps)
            else:  # if it is already BatchNormExpand, just update bf
                module.register_parameter('bf', nn.Parameter(bias_feature_maps))
            
            # Mark corresponding weights to be pruned
            module.weight.data.mul_(~module.pruned_input)
        
        # TODO this can be smart to do but atm it breaks everything
        # if getattr(module, 'bias', None) is not None and module.bias.abs().sum() == 0:
        #     module.register_parameter('bias', None)
        
        elif isinstance(module, nn.Linear):
            # TODO: handle missing bias
            # For a linear layer, we can just update the scalar bias values
            # if getattr(module, 'bias', None) is not None:
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
            output[~pruned_channels[None, :, None,
                    None].expand_as(output)] *= float('nan')
            if isinstance(module, (ConvB, ConvExpand)):
                if getattr(module, 'bf', None) is not None:
                    module.bf.data.mul_(~pruned_channels[:, None, None])
            else:
                if getattr(module, 'bias', None) is not None:
                    module.bias.data.mul_(~pruned_channels)
        
        if isinstance(module, nn.BatchNorm2d):
            output[~pruned_channels[None, :, None,
                    None].expand_as(output)] *= float('nan')
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
            handle = module.register_forward_hook(lambda m, i, o: __propagate_biases_hook(m, i, o))
            handles.append(handle)
    
    # Propagate biases
    zeros = torch.zeros_like(x)  # make sure input is zero
    model(zeros)
    
    for h in handles:
        h.remove()
    
    return model
