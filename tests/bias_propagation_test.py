import unittest

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import SqueezeNet

import simplify
import utils
from tests.benchmark_models import models
from utils import set_seed


class BiasPropagationTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)

    def test_bias_propagation(self):
        @torch.no_grad()
        def test_arch(arch, x, pretrained=False, fuse_bn=True):
            if architecture.__name__ in [
                    "shufflenet_v2_x1_5", "shufflenet_v2_x2_0", "mnasnet0_75", "mnasnet1_3"]:
                pretrained = False

            model = arch(pretrained, progress=False)
            model.eval()

            for name, module in model.named_modules():
                if isinstance(model, SqueezeNet) and 'classifier.1' in name:
                    continue

                if isinstance(module, nn.Conv2d):
                    prune.random_structured(
                        module, 'weight', amount=0.8, dim=0)
                    prune.remove(module, 'weight')

            if fuse_bn:
                bn_folding = utils.get_bn_folding(model)
                model = simplify.fuse(model, bn_folding)
            y_src = model(x)

            zeros = torch.zeros(1, *x.shape[1:])
            pinned_out = utils.get_pinned_out(model)
            simplify.propagate_bias(model, zeros, pinned_out)
            y_prop = model(x)

            #print(f'------ {self.__class__.__name__, arch.__name__} ------')
            #print("Max abs diff: ", (y_src - y_prop).abs().max().item())
            #print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
            #print(
            #    f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
            #print()

            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))

        x = torch.randint(0, 256, (256, 3, 224, 224))
        x = x.float() / 255.

        for architecture in models:
            with self.subTest(arch=architecture, pretrained=True, fuse_bn=True):
                self.assertTrue(
                    test_arch(
                        architecture,
                        x,
                        pretrained=True,
                        fuse_bn=True))

            with self.subTest(arch=architecture, pretrained=True, fuse_bn=False):
                self.assertTrue(
                    test_arch(
                        architecture,
                        x,
                        pretrained=True,
                        fuse_bn=False))
