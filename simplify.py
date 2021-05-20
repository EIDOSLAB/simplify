from conv import ConvB
import sys
import torch
import torch.nn as nn

from collections import OrderedDict
from typing import Any, List

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
def propagate_biases_hook(module, input, output):
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
        bias_feature_maps = module(input)[0] # [out_channels x W x H]; or [out_features] 

    if isinstance(module, nn.Conv2d):
        if getattr(module, 'bias', None) is not None:
            bias_feature_maps -= module.bias[:, None, None]
        module = ConvB.from_conv(module, bias_feature_maps)
    
    elif isinstance(module, nn.Linear):
        if getattr(module, 'bias', None) is not None:      
            module.bias.copy_(bias_feature_maps)

    # Step 2. Propagate output to next module
    # Zero-out everything except for biases
    output.mul_(0.).abs()

    if hasattr(module, 'bias') and module.bias is not None:
        # Compute mask of zeroed (pruned) channels
        shape = module.weight.shape
        zero_mask = module.weight.view(shape[0], -1).sum(dim=1) == 0

        # Propagate only the bias values corresponding to pruned channels
        shape = output.shape
        output.view(shape[0], shape[1], -1).add_((module.bias*zero_mask)[None, :, None])

        # Remove biases of pruned channels
        module.bias.data.mul_(~zero_mask)
    
    for output_channel in output[0]:
        assert torch.unique(output_channel).shape[0] == 1

    pass

@torch.no_grad()
def __propagate_bias(model: nn.Module, x) -> nn.Module:
    handles = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handle = module.register_forward_hook(propagate_biases_hook)
            handles.append(handle)

    # Propagate biases
    zeros = torch.zeros_like(x) #make sure input is zero
    model(zeros)

    for h in handles:
        h.remove()

    return model

def __remove_zeroed(model: nn.Module, pinned_in: List, pinned_out: List) -> nn.Module:
    """
    TODO: doc
    """
    
    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            # If not pinned_in: remove input channels corresponding to previous removed output channels
            # If not pinned_in: remove zeroed input channels
            # If not pinned_out: remove zeroed output channels
            pass
    
    return model


def simplify(model: nn.Module, x: torch.Tensor, pinned_in=None, pinned_out=None) -> nn.Module:
    if pinned_in is None:
        pinned_in = []
    
    if pinned_out is None:
        pinned_out = []
    
    model = __propagate_bias(model, x)
    model = __remove_zeroed(model, pinned_in, pinned_out)
    
    return model
