from os import error
from os import error
from typing import List

import torch
import torch.nn as nn

import fuser
from conv import ConvB, ConvExpand


@torch.no_grad()
def __propagate_bias(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:

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
        
        pruned_input = input.squeeze(dim=0)
        pruned_input = pruned_input.view(pruned_input.shape[0], -1).sum(dim=1) == 0
        
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
            # TODO: if bias is missing, it must be inserted here
            if getattr(module, 'bias', None) is not None:
                #TODO: check if bias can be != scalar ([:, 0, 0])
                module.bias[pruned_input].copy_(bias_feature_maps[:, 0, 0][pruned_input])
                module.weight.data.mul_(~pruned_input)
            # TODO: if bias is missing, it must be inserted here

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

@torch.no_grad()
def __remove_zeroed(model: nn.Module, x: torch.Tensor, pinned_out: List) -> nn.Module:
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
            #TODO: handle when groups > 1 (if possible)

        elif isinstance(module, nn.Linear):
            module.weight.data = module.weight.data[:, nonzero_idx]
            module.in_features = module.weight.shape[1]

        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data = module.weight.data[nonzero_idx]
            module.num_features = module.weight.shape[0]
        
        # Compute remaining channels indices
        output = torch.ones_like(output)
        if isinstance(module, nn.Conv2d) and module.groups > 1:
            return output
            
        # If not pinned: remove zeroed output channels
        if not isinstance(module, nn.BatchNorm2d):
            shape = module.weight.shape
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
            module = ConvExpand.from_conv(module, idxs, module.bf)
        else:
            if getattr(module, 'bias', None) is not None:
                module.bias.data = module.bias.data[nonzero_idx]
            
            if getattr(module, 'bf', None) is not None:
                module.bf.data = module.bf.data[nonzero_idx]
                
            if isinstance(module, nn.BatchNorm2d):
                module.running_mean.data = module.running_mean.data[nonzero_idx]
                module.running_var.data = module.running_var.data[nonzero_idx]
            
            output *= 0.
            output[:, nonzero_idx] = 1
        
        if isinstance(module, nn.Conv2d):
            module.out_channels = module.weight.shape[0]
        elif isinstance(module, nn.Linear):
            module.out_features = module.weight.shape[0]
        
        return output
    
    def __skip_activation_hook(module, input, output):
        return input[0]
    
    # TODO: add all activation layers
    activations = [
        nn.ReLU,
        nn.Tanh,
        nn.Sigmoid,
        nn.Hardswish,
        nn.Hardsigmoid
    ]

    handles = []
    for name, module in model.named_modules():
        # TODO: add all parameters layers
        if not isinstance(module, (*activations, nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
            continue
        
        if len(list(module.parameters())) == 0:
            # Skip activation/identity etc layers
            handle = module.register_forward_hook(__skip_activation_hook)
        else:
            handle = module.register_forward_hook(lambda m, i, o, n=name: __remove_zeroed_channels_hook(m, i, o, n))
        handles.append(handle)
    
    x = torch.ones_like(x)
    model(x)
    
    for h in handles:
        h.remove()
    
    return model


def simplify(model: nn.Module, x: torch.Tensor, pinned_out: List=None, bn_folding: List=None) -> nn.Module:
    if bn_folding is None:
        bn_folding = []
    if pinned_out is None:
        pinned_out = []

    model = fuser.convert_bn(model, bn_folding)
    __propagate_bias(model, x, pinned_out)
    __remove_zeroed(model, x, pinned_out)
    
    return model
