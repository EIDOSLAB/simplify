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
             training: bool = False, pinned_out: List = None) -> nn.Module:
    if training and fuse_bn:
        print("Cannot fuse BatchNorm in training mode")
        fuse_bn = False
    
    if fuse_bn:
        if bn_folding is None:
            bn_folding = utils.get_bn_folding(model)
        fuse(model, bn_folding)

    if pinned_out is None:    
        pinned_out = utils.get_pinned(model)
    
    propagate_bias(model, x, pinned_out)
    remove_zeroed(model, x, pinned_out)
    
    return model
