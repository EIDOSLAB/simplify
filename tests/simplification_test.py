import unittest

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models.squeezenet import SqueezeNet

import simplify
import simplify.utils as utils
from tests.benchmark_models import models
from simplify.utils import set_seed


class SimplificationTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)

    def test_simplification(self):
        @torch.no_grad()
        def test_arch(arch, x, pretrained=False, fuse_bn=True):
            if architecture.__name__ in ["shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
                pretrained = False

            model = arch(pretrained, progress=False)
            model.eval()
            
            for name, module in model.named_modules():
                if isinstance(model, SqueezeNet) and 'classifier.1' in name:
                    continue

                if isinstance(module, nn.Conv2d):
                    prune.random_structured(module, 'weight', amount=0.8, dim=0)
                    prune.remove(module, 'weight')

            y_src = model(x)
            zeros = torch.zeros(1, *x.shape[1:])

            simplify.simplify(model, zeros, fuse_bn=fuse_bn)
            y_prop = model(x)

            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))

        im = torch.randint(0, 256, (256, 3, 224, 224))
        x = im / 255.

        for architecture in models:
            with self.subTest(arch=architecture.__name__, pretrained=True, fuse_bn=True):
                self.assertTrue(test_arch(architecture, x, True, True))
            
            with self.subTest(arch=architecture.__name__, pretrained=True, fuse_bn=False):
                self.assertTrue(test_arch(architecture, x, True, False))

