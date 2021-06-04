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
def convert_bn(model, bn_folding, device="cpu"):
    for module_pair in bn_folding:
        preceding_name = module_pair[0].split(".")
        bn_name = module_pair[1].split(".")
        preceding = get_module(model, preceding_name)
        bn = get_module(model, bn_name)
        
        if isinstance(preceding, nn.Linear):
            fused_module = fuse_fc_and_bn(preceding, bn, device)
        if isinstance(preceding, nn.Conv2d):
            fused_module = fuse_conv_and_bn(preceding, bn, device)

        substitute_module(model, fused_module, preceding_name)
        substitute_module(model, nn.Identity(), bn_name)
        
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_w = module.weight.data
            bn_b = module.bias.data
            bn_m = module.running_mean.data
            bn_v = module.running_var.data
    
            conv_w = bn_w / torch.sqrt(bn_v + module.eps)
            conv_b = bn_b - (bn_w * bn_m) / torch.sqrt(bn_v + module.eps)
    
            bn_as_conv = nn.Conv2d(module.weight.shape[0], module.weight.shape[0], (1, 1))

            bn_as_conv.bias.data = conv_b
            conv_w_zero = torch.zeros_like(bn_as_conv.weight.data)
            for ch in range(conv_w.shape[0]):
                conv_w_zero[ch, ch] = conv_w[ch]
            bn_as_conv.weight.data = conv_w_zero

            substitute_module(model, bn_as_conv, name.split("."))
    
    return model


@torch.no_grad()
def fuse_conv_and_bn(conv, bn, device):
    """
    Perform modules fusion.
    :param conv: nn.Conv2d module.
    :param bn: nn.BatchNorm2d module.
    :param device: Device on which build the new nn.Conv2d module.
    :return: nn.Conv2d originated from $conv$ and $bn$ fusion.
    """
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    # init
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
    bn_diag = torch.diag(bn_weight.div(torch.sqrt(bn_eps + bn_var.to(torch.double))))
    fusedconv_weight = torch.mm(bn_diag, conv_weight).view(fusedconv.weight.size()).to(torch.float)
    fusedconv.weight.copy_(fusedconv_weight)
    
    # prepare spatial bias
    b_bn = bn_bias - bn_weight.mul(bn_mean).div(torch.sqrt(bn_var + bn_eps))
    fusedconv.bias.copy_((torch.mm(bn_diag, conv_bias.reshape(-1, 1)).reshape(-1) + b_bn).to(torch.float))
    
    return fusedconv


@torch.no_grad()
def fuse_fc_and_bn(fc, bn, device):
    fusedlinear = nn.Linear(in_features=fc.in_features, out_features=fc.out_features, bias=True)
    
    bn_weight = bn.weight.to(torch.double)
    bn_bias = bn.bias.to(torch.double)
    bn_mean = bn.running_mean.to(torch.double)
    bn_var = bn.running_var.to(torch.double)
    bn_eps = bn.eps
    fc_weight = fc.weight.to(torch.double)
    fc_bias = fc.bias.to(torch.double) if fc.bias is not None \
        else torch.zeros(fc.weight.size(0), dtype=torch.double, device=device)
    
    # prepare filters
    bn_diag = torch.diag(bn_weight.div(torch.sqrt(bn_eps + bn_var.to(torch.double))))
    fusedlinear_weight = torch.mm(bn_diag, fc_weight).view(fusedlinear.weight.size()).to(torch.float)
    fusedlinear.weight.copy_(fusedlinear_weight)
    
    # prepare spatial bias
    b_bn = bn_bias - bn_weight.mul(bn_mean).div(torch.sqrt(bn_var + bn_eps))
    fusedlinear.bias.copy_((torch.mm(bn_diag, fc_bias.reshape(-1, 1)).reshape(-1) + b_bn).to(torch.float))
    
    return fusedlinear
