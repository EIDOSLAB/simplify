from collections import OrderedDict

import torch
from EIDOSearch.models import LeNet5, LeNet300
from EIDOSearch.pruning.simplification.fuser import fuse
from torch import nn
from torchvision.models import resnet18


def zero_hook(module, input, output):
    import pydevd
    pydevd.settrace(suspend=False, trace_only_current_thread=True)
    
    print(module)
    
    input = input[0]
    tmp_hooks = module._forward_hooks
    module._forward_hooks = OrderedDict()
    biases = module(input)[0, :, module.padding[0], module.padding[1]]
    module._forward_hooks = tmp_hooks
    
    zero_mask = input[0, :, 0, 0] != 0
    module.bias.data = zero_mask * biases + (1 - zero_mask) * biases
    
    shape = module.weight.shape
    zero_mask = module.weight.view(shape[0], -1).sum(dim=1) == 0
    
    output.data.mul_(0.)
    if hasattr(module, "bias") and module.bias is not None:
        output.view(output.shape[0], output.shape[1], -1).data.add_((module.bias * zero_mask)[None, :, None])


if __name__ == '__main__':
    model = LeNet300()
    model.eval()
    model = fuse(model)
    
    dummy_input = torch.randn(1, 1, 28, 28)
    
    with torch.no_grad():
        model.fc1.weight[0] = torch.zeros_like(model.fc1.weight[0])
        model.fc1.bias[0] = 0xcafebabe
    
    for n, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            m.register_forward_hook(zero_hook)
    
    with torch.no_grad():
        y2 = model(torch.zeros(1, 1, 28, 28))
    
    print(y2)
