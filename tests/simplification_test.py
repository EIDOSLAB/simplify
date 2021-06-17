import unittest

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models.squeezenet import SqueezeNet

import simplify
import utils
from tests.benchmark_models import models
from utils import set_seed


class SimplificationTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)
    
    def test_simplification(self):
        @torch.no_grad()
        def test_arch(arch, x, pretrained=False):
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
            
            bn_folding = utils.get_bn_folding(model)
            pinned_out = utils.get_pinned_out(model)
            simplify.simplify(model, zeros, pinned_out, bn_folding)
            y_prop = model(x)
            
            print(f'------ {self.__class__.__name__, arch.__name__} ------')
            print("Max abs diff: ", (y_src - y_prop).abs().max().item())
            print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
            print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
            print()
            
            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))
        
        im = torch.randint(0, 256, (256, 3, 224, 224))
        x = im / 255.
        
        for architecture in models:
            with self.subTest(arch=architecture, pretrained=True):
                self.assertTrue(test_arch(architecture, x, True))
