import unittest
from EIDOSearch.models.architectures import LeNet5


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchvision.models.vgg import *
from torchvision.models.resnet import *

from EIDOSearch.pruning.simplification.fuser import fuse
from simplify import __propagate_bias, no_forward_hooks

@torch.no_grad()
def test_arch(arch, x):
    model = arch(pretrained=True)
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.5, dim=0)
            prune.remove(module, 'weight')

    model = fuse(model)
    y_src = model(x)

    zeros = torch.zeros_like(x)
    __propagate_bias(model, zeros)
    y_prop = model(x)
    
    print("Max abs diff: ", (y_src - y_prop).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())

    return torch.allclose(y_src, y_prop)

class HooksCtxTest(unittest.TestCase):
    def test_hook_ctx(self):
        def hook(*args):
            pass
        
        model = resnet18(pretrained=False)
        for module in model.modules():
            module.register_forward_hook(hook)
        
        model_hooks = model._forward_hooks

        with no_forward_hooks(model):
            self.assertEqual(len(model._forward_hooks), 0)
        
        self.assertEqual(model._forward_hooks, model_hooks)

class BiasPropagationTest(unittest.TestCase):
    def test_bias_propagation(self):
        x = torch.randn((1, 3, 224, 224))

        #test_arch(vgg16, x)
        #test_arch(vgg16_bn, x)
        self.assertTrue(test_arch(resnet18, x))
        #test_arch(resnet34, x)
        #test_arch(resnet50, x)
        #test_arch(resnet101, x)

if __name__ == '__main__':
    unittest.main()