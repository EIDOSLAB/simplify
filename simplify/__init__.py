from typing import List

import torch
import torch.nn as nn

from . import layers
from . import utils
from .fuse import fuse
from .propagate import propagate_bias
from .remove import remove_zeroed

__version__ = "1.0.1"


def simplify(model: nn.Module, x: torch.Tensor, fuse_bn=True, bn_folding: List = None) -> nn.Module:
    
    if fuse_bn:
        if bn_folding is None:
            bn_folding = utils.get_bn_folding(model)
        
        fuse(model, bn_folding)

    pinned_out = utils.get_pinned_out(model)
    propagate_bias(model, x, pinned_out)
    remove_zeroed(model, x, pinned_out)

    return model
