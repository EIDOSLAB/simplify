import unittest

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision
import simplify

from simplify.layers import ConvB, ConvExpand
from utils import set_seed


class ConvBTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)

    def test_conv_b(self):
        conv = nn.Conv2d(
            3,
            64,
            3,
            1,
            padding=2,
            padding_mode='zeros',
            bias=True)
        out1 = conv(torch.zeros((1, 3, 128, 128)))

        bias = conv.bias.data.clone()
        conv.bias.data.mul_(0)

        conv = ConvB.from_conv(conv, bias[:, None, None].expand_as(out1[0]))
        out2 = conv(torch.zeros((1, 3, 128, 128)))
        out2.sum().backward()

        assert(conv.weight.grad.shape == conv.weight.shape)


class ConvExpandTest(unittest.TestCase):
    # TODO update __repr__ with correct output size
    def setUp(self):
        set_seed(3)

    @torch.no_grad()
    def test_expansion(self):
        module = nn.Conv2d(3, 64, 3, 1, padding=1, bias=False)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

        x = torch.randn((57, 3, 128, 128))

        prune.random_structured(module, 'weight', amount=0.5, dim=0)
        prune.remove(module, 'weight')

        y_src = module(x)

        shape1 = module.weight.shape
        nonzero_idx = ~(module.weight.sum(dim=(1, 2, 3)) == 0)
        module.weight.data = module.weight.data[nonzero_idx]
        shape2 = module.weight.shape
        self.assertFalse(shape1 == shape2)

        y_post = module(x)
        self.assertFalse(torch.equal(y_src, y_post))

        module = ConvB.from_conv(module, torch.zeros_like(y_post)[0])
        module.register_parameter('bias', None)

        idxs = torch.where(nonzero_idx)[0]
        module = ConvExpand.from_conv(module, idxs, torch.zeros_like(y_src)[0])
        module.out_channels = module.weight.shape[0]

        for i in range(2):
            with torch.enable_grad():
                y_post = module(x)
                y_post.sum().backward()
                optimizer.step()
                optimizer.zero_grad()

        assert(module.weight.grad.shape == module.weight.shape)

    def test_model(self):
        model = torchvision.models.resnet50(True)
        x = torch.randn((57, 3, 128, 128)) 

        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                prune.random_structured(module, 'weight', amount=0.8, dim=0)
                prune.remove(module, 'weight')

    
        zeros = torch.zeros(1, *x.shape[1:])
        model.eval()
        simplify.simplify(model, zeros, fuse_bn=False)
        model.train()
        
        with torch.enable_grad():
            y = model(x)
            y.sum().backward()

        for p in model.parameters():
            self.assertIsNotNone(p.weight.grad)
            self.assertTrue(p.weight.shape == p.weight.grad.shape)