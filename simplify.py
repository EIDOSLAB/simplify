import torch.nn as nn


def __propagate_bias(model: nn.Module, pinned_out) -> nn.Module:
    return model


def __remove_zeroed(model: nn.Module, pinned_in, pinned_out) -> nn.Module:
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


def simplify(model: nn.Module, pinned_in=None, pinned_out=None) -> nn.Module:
    if pinned_in is None:
        pinned_in = []
    
    if pinned_out is None:
        pinned_out = []
    
    model = __propagate_bias(model, pinned_out)
    model = __remove_zeroed(model, pinned_in, pinned_out)
    
    return model
