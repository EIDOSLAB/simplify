from collections import OrderedDict
from os import error
from typing import Any, List

import torch
import torch.nn as nn

import fuser
from conv import ConvB, ConvExpand


# This is weird, IDK
class no_forward_hooks():
    """
    Context manager to temporarily disable forward hooks
    when execting a forward() inside a hook (i.e. avoid
    recursion)
    """
    
    def __init__(self, module: nn.Module):
        self.module = module
        self.hooks = module._forward_hooks
    
    def __enter__(self) -> None:
        self.module._forward_hooks = OrderedDict()
    
    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.module._forward_hooks = self.hooks


@torch.no_grad()
def __propagate_bias(model: nn.Module, x, pinned_out: List) -> nn.Module:

    @torch.no_grad()
    def __propagate_biases_hook(module, input, output, pinned=False):
        """
        Parameters:
            - module: nn.module
            - input: torch.Tensor; non-zero channels correspond to remaining biases pruned channels in the previos module,
                    and thus should be used to update the current biases
            - output: torch.Tensor
        """
        input = input[0]
        
        # Step 1. Fuse biases of pruned channels in the previous module into the current module
        with no_forward_hooks(module):
            bias_feature_maps = module(input)[0]  # [out_channels x W x H]; or [out_features]
        
        if isinstance(module, nn.Conv2d):
            assert module.dilation[0] == 1

            if getattr(module, 'bias', None) is not None:
                module.bias.data.mul_(0).abs_()
                #bias_feature_maps -= module.bias[:, None, None]
            module = ConvB.from_conv(module, bias_feature_maps)
        
        elif isinstance(module, nn.Linear):
            if getattr(module, 'bias', None) is not None:
                module.bias.copy_(bias_feature_maps)
        
        # Step 2. Propagate output to next module
        # Zero-out everything except for biases
        output.mul_(0.).abs_()
        # if pinned:
        #     return
        
        if hasattr(module, 'bias') and module.bias is not None:
            # Compute mask of zeroed (pruned) channels
            shape = module.weight.shape
            zero_mask = module.weight.view(shape[0], -1).sum(dim=1) == 0
            
            # Propagate only the bias values corresponding to pruned channels
            if isinstance(module, nn.Linear):
                shape = output.shape
                output.view(shape[0], shape[1], -1).add_((module.bias * zero_mask)[None, :, None])
                # Remove biases of pruned channels
                module.bias.data.mul_(~zero_mask)

            elif isinstance(module, ConvB):
                output.add_((module.bf * zero_mask[:, None, None])[None, :])
                module.bf.data.mul_(~zero_mask[:, None, None])
            
            else:
                error(module)
        
        for output_channel in output[0]:
            assert torch.unique(output_channel).shape[0] == 1

    handles = []
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            pinned = name in pinned_out
            handle = module.register_forward_hook(lambda m, i, o, p=pinned: __propagate_biases_hook(m, i, o, p))
            handles.append(handle)
    
    # Propagate biases
    zeros = torch.zeros_like(x)  # make sure input is zero
    model(zeros)
    
    for h in handles:
        h.remove()
    
    return model


def __remove_zeroed(model: nn.Module, pinned_out: List) -> nn.Module:
    """
    TODO: doc
    """
    
    def __remove_zeroed_channels_hook(module, input, output, pinned):
        """
            input: idx of previously remaining channels
        """
        input = input[0][0]  # get first item of batch
        
        # Remove input channels
        nonzero_idx = ~(input.view(input.shape[0], -1).sum(dim=1) == 0)
        module.weight.data = module.weight.data[:, nonzero_idx]
        
        if isinstance(module, nn.Conv2d):
            module.in_channels = module.weight.shape[1]
        elif isinstance(module, nn.Linear):
            module.in_features = module.weight.shape[1]
        
        # Compute remaining channels indices
        output.data = torch.ones_like(output)
        # if pinned:
        #     return
        
        # If not pinned: remove zeroed output channels
        shape = module.weight.shape
        nonzero_idx = ~(module.weight.view(shape[0], -1).sum(dim=1) == 0)
        module.weight.data = module.weight.data[nonzero_idx]
        
        if getattr(module, 'bias', None) is not None:
            module.bias.data = module.bias.data[nonzero_idx]
        
        if getattr(module, 'bf', None) is not None:
            module.bf.data = module.bf.data[nonzero_idx]
        
        if pinned:
            idxs = []
            current = 0
            zero_idx = torch.where(~nonzero_idx)[0]
            for i in range(module.weight.data.shape[0] + len(zero_idx)):
                if i in zero_idx:
                    idxs.append(module.weight.data.shape[0])
                else:
                    idxs.append(current)
                    current += 1
            module = ConvExpand.from_conv(module, idxs)
        else:
            output.data.mul_(0)
            output.data[:, nonzero_idx] = 1
        
        if isinstance(module, nn.Conv2d):
            module.out_channels = module.weight.shape[0]
        elif isinstance(module, nn.Linear):
            module.out_features = module.weight.shape[0]
        
        pass
    
    def __skip_activation_hook(module, input, output):
        output.data = input[0].data
    
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, (nn.ReLU, nn.Linear, nn.Conv2d)):
            continue
        
        if len(list(module.parameters())) == 0:
            # Skip activation/identity etc layers
            handle = module.register_forward_hook(__skip_activation_hook)
        else:
            pinned = name in pinned_out
            handle = module.register_forward_hook(lambda m, i, o, p=pinned: __remove_zeroed_channels_hook(m, i, o, p))
        handles.append(handle)
    
    x = torch.ones((1, 3, 224, 224))
    model(x)
    
    for h in handles:
        h.remove()
    
    return model


def simplify(model: nn.Module, x: torch.Tensor, pinned_out=None) -> nn.Module:
    if pinned_out is None:
        pinned_out = []
        
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = False

    model = fuser.fuse(model)
    __propagate_bias(model, x, pinned_out)
    __remove_zeroed(model, pinned_out)
    
    return model
