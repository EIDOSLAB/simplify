import unittest
from EIDOSearch.models.architectures import LeNet5


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchvision.models.alexnet import *
from torchvision.models.vgg import *
from torchvision.models.resnet import *

from EIDOSearch.pruning.simplification.fuser import fuse
from simplify import __propagate_bias, no_forward_hooks

@torch.no_grad()
def test_arch(arch, x, pretrained=None):
    model = arch() if pretrained is None else arch(pretrained)
    model.eval()

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            prune.random_structured(module, 'weight', amount=0.3, dim=0)
            prune.remove(module, 'weight')

    model = fuse(model)
    y_src = model(x)

    zeros = torch.zeros_like(x)
    __propagate_bias(model, zeros)
    y_prop = model(x)

    print("Max abs diff: ", (y_src - y_prop).abs().max().item())
    print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())

    return torch.allclose(y_src, y_prop)

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
    def test_bias_propagation(self):
        x = torch.randn((1, 3, 224, 224))

        self.assertTrue(test_arch(LeNet5, torch.randn((1, 1, 28, 28))))
        self.assertTrue(test_arch(alexnet, x, False))
        self.assertTrue(test_arch(alexnet, x, True))
        self.assertTrue(test_arch(vgg16, x, True))
        self.assertTrue(test_arch(vgg16_bn, x, True))
        self.assertTrue(test_arch(resnet18, x, True))
        self.assertTrue(test_arch(resnet34, x, True))
        self.assertTrue(test_arch(resnet50, x, True))
        self.assertTrue(test_arch(resnet101, x, True))

class PaddingTest(unittest.TestCase):
    def test_disable_padding(self):
        for stride in [1,2,3,4]:
            for padding in [1,2,3,4,5]:
                conv_zeros = nn.Conv2d(3, 64, 3, stride, padding=padding, padding_mode='zeros', bias=True)
                conv_reflect = nn.Conv2d(3, 64, 3, stride, padding=padding, padding_mode='reflect', bias=True)

                conv_reflect.weight.data.copy_(conv_zeros.weight.data)
                conv_reflect.bias.data.copy_(conv_zeros.bias.data)

                x = torch.randn((1, 3, 128, 128))

                y_reflect = conv_reflect(x)

                conv_zeros.padding_mode = 'reflect'
                y_zeros = conv_zeros(x)
                
                self.assertTrue(torch.equal(y_reflect, y_zeros))

class BiasTest(unittest.TestCase):
    def test_disable_padding(self):
        conv = nn.Conv2d(3, 64, 3, 1, padding=2, padding_mode='zeros', bias=True)

        x = torch.randn((1, 3, 128, 128))
        y1 = conv(x)

        print('y1.shape', y1.shape)
        print('bias.shape', conv.bias.shape)
        print('expanded bias.shape', conv.bias.data[:, None, None].shape, conv.bias.data[:, None, None].expand_as(y1[0]).shape)
        conv.bias.data = conv.bias.data[:, None, None].expand_as(y1[0])
        y2 = conv(x) 
        
        self.assertTrue(torch.equal(y1, y2))



if __name__ == '__main__':
    unittest.main()