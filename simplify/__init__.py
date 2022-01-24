#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).

from typing import List

import torch
import torch.nn as nn

from . import layers
from . import utils
from .fuse import fuse
from .propagate import propagate_bias
from .remove import remove_zeroed

__version__ = "1.1.0"


def simplify(model: nn.Module, x: torch.Tensor, bn_folding: List = None, fuse_bn: bool = True,
             pinned_out: List = None) -> nn.Module:
    """
    Main method. Is the standard simplification procedure that automatically calls all the necessary sub-procedures.
    It currently supports all the existing PyTorch classification models.

    Args:
        model (torch.nn.Module): Module to be simplified.
        x (torch.Tensor): `model`'s input of shape [1, C, N, M], same as the model usual input.
        bn_folding (List): List of tuple (`nn.Conv2d`, `nn.BatchNorm2d`) to be fused. If None it tries to evaluate them given the model. Default `None`.
        fuse_bn (bool): If True, fuse the conv-bn tuple.
        pinned_out (List): List of `nn.Modules` which output needs to remain of the original shape (e.g. layers related to a residual connection with a sum operation).

    Returns:
        torch.nn.Module: Simplified model.

    """
    
    if fuse_bn:
        if bn_folding is None:
            bn_folding = utils.get_bn_folding(model)
        fuse(model, bn_folding)

    if pinned_out is None:    
        pinned_out = utils.get_pinned(model)
    
    propagate_bias(model, x, pinned_out)
    remove_zeroed(model, x, pinned_out)
    
    return model
