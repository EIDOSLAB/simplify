from conv import ConvB
import unittest
import random

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import os
import copy

from torchvision.models.alexnet import *
from torchvision.models.vgg import *
from torchvision.models.resnet import *

from simplify import __propagate_bias as propagate, no_forward_hooks
from fuser import fuse

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

@torch.no_grad()
def test_arch(arch, x, pretrained=False):
    model = arch(pretrained, progress=False)
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.8, dim=0)
            prune.remove(module, 'weight')

    model = fuse(model)
    y_src = model(x)

    zeros = torch.zeros(1, *x.shape[1:])
    propagate(model, zeros)
    y_prop = model(x)

    print(f'------ {arch} ------')
    print("Max abs diff: ", (y_src - y_prop).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
    print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
    print()
        
    return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))

class ZeroHooksTest(unittest.TestCase):
    def test_overwrite_output(self):
        def hook(m, i, output):
            output.data.mul_(0)

        conv1 = nn.Conv2d(3, 64, 3, 3, padding=2, padding_mode='zeros', bias=True)
        conv2 = nn.Conv2d(64, 3, 3, 3, padding=2, padding_mode='zeros', bias=True)
        seq = torch.nn.Sequential(conv1, conv2)

        x = torch.randn((1, 3, 128, 128))

        no_hook_conv = conv1(x)
        no_hook = seq(x)

        conv1.register_forward_hook(hook)
        w_hook_conv = conv1(x)
        w_hook = seq(x)
        
        self.assertNotEqual(no_hook_conv.abs().sum(), 0)
        self.assertEqual(w_hook_conv.sum(), 0)
        self.assertFalse(torch.equal(no_hook, w_hook))
   
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
    
    @torch.no_grad()
    def test_conv_manual_bias_float32(self):
        module = nn.Conv2d(3, 64, 3, padding=1)
        x = torch.randn((64, 3, 128, 128))

        y_src = module(x)

        bias = module.bias.data.clone()
        module.bias.data.mul_(0)
        y_prop = module(x) + bias[:, None, None]

        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

    @torch.no_grad()
    @unittest.skip
    def test_conv_manual_bias_float64(self):
        module = nn.Conv2d(3, 64, 3, padding=1).double()
        x = torch.randn((64, 3, 128, 128)).double()

        y_src = module(x)

        bias = module.bias.data.clone()
        module.bias.data.mul_(0)
        y_prop = module(x) + bias[:, None, None]

        self.assertTrue(torch.equal(y_src, y_prop))

    @torch.no_grad()
    def test_linear(self):
        model = nn.Sequential(nn.Linear(128, 256),
                              nn.ReLU(),
                              nn.Linear(256, 512),
                              nn.ReLU(),
                              nn.Linear(512, 10))

        for module in list(model.children())[:-1]:
            if isinstance(module, nn.Linear):
                prune.random_structured(module, 'weight', amount=0.1, dim=0)
                prune.remove(module, 'weight')
    
        x = torch.randn(512, 128)
        zeros = torch.zeros(1, 128)
        
        y_src = model(x)
        propagate(model, zeros)
        y_prop = model(x)

        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

    @torch.no_grad()
    def test_conv(self):
        model = nn.Sequential(nn.Conv2d(3, 64, 3, padding=2),
                              nn.ReLU(), 
                              nn.Conv2d(64, 128, 3, padding=5),
                              nn.ReLU(),
                              nn.Conv2d(128, 32, 11, padding=7))

        for module in list(model.children())[:-1]:
            if isinstance(module, nn.Conv2d):
                prune.random_structured(module, 'weight', amount=0.9, dim=0)
                prune.remove(module, 'weight')

        x = torch.randn(128, 3, 128, 128)
        zeros = torch.zeros(1, 3, 128, 128)

        y_src = model(x)
        model = propagate(model, zeros)
        y_prop = model(x)

        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

    def test_bias_propagation(self):
        #x = torch.randn((32, 3, 224, 224))
        x = torch.randint(0, 256, ((256, 3, 224, 224)))
        x = x.float()/255.

        test_idx = 0
        for architecture in [alexnet, vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet101]:
            with self.subTest(i=test_idx, arch=architecture, pretrained=True):
                self.assertTrue(test_arch(architecture, x, True))
            test_idx += 1

class ConvBTest(unittest.TestCase):
    def test_conv_b(self):
        conv = nn.Conv2d(3, 64, 3, 1, padding=2, padding_mode='zeros', bias=True)
        out1 = conv(torch.zeros((1, 3, 128, 128)))
        
        bias = conv.bias.data.clone()
        conv.bias.data.mul_(0)

        conv = ConvB.from_conv(conv, bias[:, None, None].expand_as(out1[0]))
        out2 = conv(torch.zeros((1, 3, 128, 128)))

        self.assertTrue(torch.equal(out1, out2))
       
if __name__ == '__main__':
    torch.set_printoptions(precision=10)
    set_seed(0)
    unittest.main()