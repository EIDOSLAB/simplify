import sys
from collections import OrderedDict

import torch
from EIDOSearch.models import LeNet5
from EIDOSearch.pruning.simplification.fuser import fuse
from torch import nn


def forward_with_no_hooks(module, input):
    # This is weird, IDK
    tmp_hooks = module._forward_hooks
    module._forward_hooks = OrderedDict()
    biases = module(input)[0, :, module.padding[0], module.padding[1]] if isinstance(module, nn.Conv2d) \
        else module(input)[0, :]
    module._forward_hooks = tmp_hooks
    
    return biases


def check(module, biases, input):
    print(module)
    print("Current biases")
    print(module.bias)
    print("New biases")
    print(biases)
    ##############
    # For linear #
    ##############
    if isinstance(module, nn.Linear):
        manual_biases = input * module.weight
        update = torch.sum(manual_biases, dim=1)
    ############
    # For conv #
    ############
    if isinstance(module, nn.Conv2d):
        update = torch.zeros(module.weight.shape[0])
        for i in range(input.shape[1]):
            manual_biases = module.weight[:, i, :, :].mul(input[0, :, module.padding[0], module.padding[1]][i])
            update += torch.sum(torch.sum(manual_biases, dim=2), dim=1)
    print("Manual update (old)")
    print(update)
    print("Manually updated biases (old)")
    print(module.bias + update)
    print("Difference")
    print((module.bias + update) - biases)


@torch.no_grad()
def zero_hook(module, input, output):
    if sys.gettrace() is not None:  # PyCharm debugger for hooks
        import pydevd
        pydevd.settrace(suspend=False, trace_only_current_thread=True)
    
    """
    Uncomment `check(module, biases, input)` below to check that:
    - `module.bias + update` should be equal to `biases`
    - if input is 0, `update` should be 0 and `biases` should be equal to `module.bias`
    - if input is NOT 0, `update` should be NON 0 and `biases` should be different than `module.bias`
    """
    
    input = input[0]
    biases = forward_with_no_hooks(module, input)  # This are the new biases for this module
    
    # check(module, biases, input)
    
    module.bias.copy_(biases)  # Update the module's biases
    
    shape = module.weight.shape
    zero_mask = module.weight.view(shape[0], -1).sum(dim=1) == 0
    
    output.mul_(0.)
    if hasattr(module, "bias") and module.bias is not None:
        output.view(output.shape[0], output.shape[1], -1).add_((module.bias * zero_mask)[None, :, None])


if __name__ == '__main__':
    model = LeNet5()
    model.eval()
    model = fuse(model)
    
    input_shape = (1, 1, 28, 28)
    
    dummy_input = torch.randn(input_shape)
    
    with torch.no_grad():
        model.conv1.weight[0] = torch.zeros_like(model.conv1.weight[0])
        model.conv1.bias[0] = 2.
        model.fc1.weight[3] = torch.zeros_like(model.fc1.weight[0])
        model.fc1.bias[3] = 4.
    
    base_out = model(dummy_input)  # Print of "sparse-but-with-biases" model's output
    
    handles = []
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            handle = m.register_forward_hook(zero_hook)
            handles.append(handle)
    
    model(torch.zeros(input_shape))
    
    for h in handles:
        h.remove()
    
    with torch.no_grad():  # Remove the bias for the pruned neuron to simulate the simplification procedure
        model.conv1.bias[0] = 0.
        model.fc1.bias[3] = 0.
    
    prop_out = model(dummy_input)  # Print of "updated" model's output, should be equal to the previous output
    
    print("Max abs diff: ", (base_out - prop_out).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(base_out, prop_out).item())
    print(base_out)
    print(prop_out)
