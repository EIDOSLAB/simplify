#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).

import torch
import torch.nn as nn


@torch.no_grad()
def get_module(model, name):
    module = model
    for idx, sub in enumerate(name):
        if idx < len(name):
            module = getattr(module, sub)
    
    return module


@torch.no_grad()
def substitute_module(model, new_module, sub_module_names):
    """
    Substitute a nn.module in a given PyTorch model with another.
    :param model: PyTorch model on which the substitution occurs.
    :param new_module: New module to insert in the model.
    :param sub_module_names: List of string representing the old module name.
    i.e if the module name is layer1.0.conv1 `sub_module_names` should be ["layer1", "0", "conv1"]
    """
    if new_module is not None:
        attr = model
        for idx, sub in enumerate(sub_module_names):
            if idx < len(sub_module_names) - 1:
                attr = getattr(attr, sub)
            else:
                setattr(attr, sub, new_module)


@torch.no_grad()
def fuse(model, bn_folding):
    for module_pair in bn_folding:
        fused_module = None
        
        preceding_name = module_pair[0].split(".")
        bn_name = module_pair[1].split(".")
        preceding = get_module(model, preceding_name)
        bn = get_module(model, bn_name)
        
        if isinstance(bn, nn.BatchNorm2d):
            if isinstance(preceding, nn.Linear):
                fused_module = fuse_fc_and_bn(preceding, bn)
            if isinstance(preceding, nn.Conv2d):
                fused_module = fuse_conv_and_bn(preceding, bn)
        
        if fused_module is not None:
            substitute_module(model, fused_module, preceding_name)
            substitute_module(model, nn.Identity(), bn_name)
    
    return model


@torch.no_grad()
def fuse_conv_and_bn(conv, bn):
    """
    Perform modules fusion.
    :param conv: nn.Conv2d module.
    :param bn: nn.BatchNorm2d module.
    :return: nn.Conv2d originated from $conv$ and $bn$ fusion.
    """
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    # init
    device = conv.weight.device
    
    fusedconv = nn.Conv2d(in_channels=conv.in_channels,
                          out_channels=conv.out_channels,
                          kernel_size=conv.kernel_size,
                          stride=conv.stride,
                          padding=conv.padding,
                          dilation=conv.dilation,
                          groups=conv.groups,
                          bias=True,
                          padding_mode=conv.padding_mode).to(device)
    
    bn_weight = bn.weight.to(torch.double)
    bn_bias = bn.bias.to(torch.double)
    bn_mean = bn.running_mean.to(torch.double)
    bn_var = bn.running_var.to(torch.double)
    bn_eps = bn.eps
    conv_weight = conv.weight.view(conv.out_channels, -1).to(torch.double)
    conv_bias = conv.bias.to(torch.double) if conv.bias is not None \
        else torch.zeros(conv.weight.size(0), dtype=torch.double, device=device)
    
    # prepare filters
    bn_diag = torch.diag(
        bn_weight.div(
            torch.sqrt(
                bn_eps +
                bn_var.to(
                    torch.double))))
    fusedconv_weight = torch.mm(
        bn_diag,
        conv_weight).view(
        fusedconv.weight.size()).to(
        torch.float)
    fusedconv.weight.copy_(fusedconv_weight)
    
    # prepare spatial bias
    b_bn = bn_bias - bn_weight.mul(bn_mean).div(torch.sqrt(bn_var + bn_eps))
    fusedconv.bias.copy_(
        (torch.mm(bn_diag, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn).to(torch.float))
    
    return fusedconv


@torch.no_grad()
def fuse_fc_and_bn(fc, bn):
    device = fc.weight.device
    
    fusedlinear = nn.Linear(
        in_features=fc.in_features,
        out_features=fc.out_features,
        bias=True).to(device)
    
    bn_weight = bn.weight.to(torch.double)
    bn_bias = bn.bias.to(torch.double)
    bn_mean = bn.running_mean.to(torch.double)
    bn_var = bn.running_var.to(torch.double)
    bn_eps = bn.eps
    fc_weight = fc.weight.to(torch.double)
    fc_bias = fc.bias.to(torch.double) if fc.bias is not None \
        else torch.zeros(fc.weight.size(0), dtype=torch.double, device=device)
    
    # prepare filters
    bn_diag = torch.diag(
        bn_weight.div(
            torch.sqrt(
                bn_eps +
                bn_var.to(
                    torch.double))))
    fusedlinear_weight = torch.mm(
        bn_diag,
        fc_weight).view(
        fusedlinear.weight.size()).to(
        torch.float)
    fusedlinear.weight.copy_(fusedlinear_weight)
    
    # prepare spatial bias
    b_bn = bn_bias - bn_weight.mul(bn_mean).div(torch.sqrt(bn_var + bn_eps))
    fusedlinear.bias.copy_(
        (torch.mm(bn_diag, fc_bias.reshape(-1, 1)).reshape(-1) + b_bn).to(torch.float))
    
    return fusedlinear
