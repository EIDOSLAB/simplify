import unittest
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

import utils
import simplify
from utils import set_seed

class IntegrationTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)

    @torch.no_grad()
    def test_cat_residual_conv(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.module0 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
                self.module1 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.module2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.module3 = nn.Conv2d(64*2, 64, 3, stride=1, padding=1, bias=True)
                self.relu = nn.ReLU()
                
                prune.random_structured(self.module0, 'weight', amount=0.5, dim=0)
                prune.remove(self.module0, 'weight')
                
                prune.random_structured(self.module1, 'weight', amount=0.4, dim=0)
                prune.remove(self.module1, 'weight')
                
                prune.random_structured(self.module2, 'weight', amount=0.8, dim=0)
                prune.remove(self.module2, 'weight')
            
                self.a = None
                self.b = None
                self.c = None
            
            def forward(self, x):
                x = self.relu(self.module0(x))
                self.a = self.module1(x)
                self.b = self.module2(x)
                self.c = torch.cat((self.a, self.b), dim=1)
                return self.module3(self.relu(self.c))
        
        residual = Residual()
        x = torch.randn((10, 3, 224, 224))
        
        y_src = residual(x)
        simplify.simplify(residual, torch.zeros((1, 3, 224, 224)), [])
        y_prop = residual(x)
        
        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

    def test_simplification(self):
        @torch.no_grad()
        def test_arch(arch, x, pretrained=False):
            model = arch(pretrained, progress=False)
            model.eval()
            pinned_out = utils.get_pinned_out(model)

            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    prune.random_structured(module, 'weight', amount=0.5, dim=0)
                    prune.remove(module, 'weight')
            
            y_src = model(x)
            zeros = torch.zeros(1, *x.shape[1:])
            
            simplify.simplify(model, zeros, pinned_out)
            y_prop = model(x)
            
            print(f'------ {self.__class__.__name__, arch.__name__} ------')
            print("Max abs diff: ", (y_src - y_prop).abs().max().item())
            print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
            print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
            print()
            
            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))
        
        im = torch.randint(0, 256, ((256, 3, 224, 224)))
        x = im / 255.
        
        for architecture in [resnet18, resnet34, resnet50, resnet101, resnet152]:
            with self.subTest(arch=architecture, pretrained=True):
                self.assertTrue(test_arch(architecture, x, True))