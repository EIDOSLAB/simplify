import unittest
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18

from simplify import no_forward_hooks
from utils import set_seed

@unittest.skip
class ZeroHooksTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)

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


@unittest.skip
class HooksCtxTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)
        
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