import copy
import unittest

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU6
import torch.nn.utils.prune as prune

import simplify
from utils import set_seed


class MiscTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)
    
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
        simplify.propagate_bias(model, zeros, [])
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
        
        x = torch.randn(10, 3, 128, 128)
        zeros = torch.zeros(1, 3, 128, 128)
        
        y_src = model(x)
        model = simplify.propagate_bias(model, zeros, [])
        y_prop = model(x)
        
        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))
    
    @torch.no_grad()
    def test_residual_skip_conv(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.module0 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
                self.module1 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
                self.module2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
                self.module3 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.relu = nn.ReLU()
                
                prune.random_structured(self.module0, 'weight', amount=0.5, dim=0)
                prune.remove(self.module0, 'weight')
                
                prune.random_structured(self.module1, 'weight', amount=0.5, dim=0)
                prune.remove(self.module1, 'weight')
                
                prune.random_structured(self.module2, 'weight', amount=0.8, dim=0)
                prune.remove(self.module2, 'weight')
                
                # self.module1.weight.data[0:32].mul_(0)
                # self.module2.weight.data[0:32].mul_(0)
                
                self.a = None
                self.b = None
                self.c = None
            
            def forward(self, x):
                x = self.relu(self.module0(x))
                residual = x
                self.a = self.module1(x)
                self.b = self.module2(self.a)
                self.c = self.b + residual
                return self.module3(self.relu(self.c))
        
        src = Residual()
        residual = copy.deepcopy(src)
        x = torch.randn((10, 3, 128, 128))
        
        y_src = residual(x)
        simplify.propagate_bias(residual, torch.zeros((1, 3, 128, 128)), {})
        y_prop = residual(x)
        self.assertFalse(torch.allclose(y_src, y_prop, atol=1e-6))
        
        residual = copy.deepcopy(src)
        y_src = residual(x)
        pinned = {
            'module2': [residual.module0]
        }
        simplify.propagate_bias(residual, torch.zeros((1, 3, 128, 128)), pinned)
        y_prop = residual(x)
        self.assertFalse(torch.allclose(y_src, y_prop, atol=1e-6))
        
        residual = copy.deepcopy(src)
        y_src = residual(x)
        pinned = {
            'module0': [residual.module2],
            'module2': [residual.module0]
        }
        simplify.propagate_bias(residual, torch.zeros((1, 3, 128, 128)), pinned)
        y_prop = residual(x)
        
        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))
    
    @torch.no_grad()
    def test_residual_conv(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.module0 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
                self.module1 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.module2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.module3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
                self.relu = nn.ReLU()
                
                prune.random_structured(self.module0, 'weight', amount=0.5, dim=0)
                prune.remove(self.module0, 'weight')
                
                prune.random_structured(self.module1, 'weight', amount=0.5, dim=0)
                prune.remove(self.module1, 'weight')
                
                prune.random_structured(self.module2, 'weight', amount=0.8, dim=0)
                prune.remove(self.module2, 'weight')
                
                # self.module1.weight.data[0:32].mul_(0)
                # self.module2.weight.data[0:32].mul_(0)
                
                self.a = None
                self.b = None
                self.c = None
            
            def forward(self, x):
                x = self.relu(self.module0(x))
                self.a = self.module1(x)
                self.b = self.module2(x)
                self.c = self.a + self.b
                return self.module3(self.relu(self.c))
        
        residual = Residual()
        x = torch.randn((10, 3, 128, 128))
        
        y_src = residual(x)
        
        pinned = {
            'module1': [residual.module2],
            'module2': [residual.module1]
        }
        simplify.propagate_bias(residual, torch.zeros((10, 3, 128, 128)), pinned)
        y_prop = residual(x)
        
        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))
    
    @torch.no_grad()
    def test_residual_linear(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.module0 = nn.Linear(10, 5, bias=True)
                self.module1 = nn.Linear(5, 3, bias=True)
                self.module2 = nn.Linear(5, 3, bias=True)
                self.module3 = nn.Linear(3, 2, bias=True)
                self.relu = nn.ReLU()
                
                prune.random_structured(self.module0, 'weight', amount=0.5, dim=0)
                prune.remove(self.module0, 'weight')
                
                prune.random_structured(self.module1, 'weight', amount=0.5, dim=0)
                prune.remove(self.module1, 'weight')
                
                prune.random_structured(self.module2, 'weight', amount=0.8, dim=0)
                prune.remove(self.module2, 'weight')
                # self.module1.weight.data[2:3].mul_(0)
                # self.module2.weight.data[2:3].mul_(0)
                
                self.a = None
                self.b = None
                self.c = None
            
            def forward(self, x):
                x = self.relu(self.module0(x))
                self.a = self.module1(x)
                self.b = self.module2(x)
                self.c = self.a + self.b
                return self.module3(self.relu(self.c))
                # return self.module3(self.c)
        
        residual = Residual()
        residual.eval()
        x = torch.randn((7, 10))
        
        y_src = residual(x)
        
        pinned = {
            'module1': [residual.module2],
            'module2': [residual.module1]
        }
        simplify.propagate_bias(residual, torch.zeros((1, 10)), pinned)
        y_prop = residual(x)
        
        print('abs max:', (y_src - y_prop).abs().max())
        
        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))
    
    @torch.no_grad()
    def test_cat_residual_conv(self):
        class Residual(nn.Module):
            def __init__(self):
                super().__init__()
                self.module0 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
                self.module1 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.module2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
                self.module3 = nn.Conv2d(64 * 2, 64, 3, stride=1, padding=1, bias=True)
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
        simplify.propagate_bias(residual, torch.zeros((1, 3, 224, 224)), [])
        y_prop = residual(x)
        
        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

    @torch.no_grad()
    def test_conv_grouping(self):
        model = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 64, 3, 1, 1, groups=64, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(64, 32, 5, 1, 1, groups=2, bias=True))

        for module in list(model.modules())[:-1]:
            if isinstance(module, nn.Conv2d):
                prune.random_structured(module, 'weight', amount=0.8, dim=0)
                prune.remove(module, 'weight')

        x = torch.randn((100, 3, 224, 224))
        y_src = model(x)

        simplify.propagate_bias(model, torch.zeros((1, 3, 224, 224)), [])
        y_prop = model(x)

        self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

    @torch.no_grad()
    def test_conv_dilation(self):
        model = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(128, 64, 3, 1, 1, dilation=2, bias=True),
                              nn.ReLU(),
                              nn.Conv2d(64, 32, 5, 1, 1, dilation=1, bias=True))

        for module in list(model.modules())[:-1]:
            if isinstance(module, nn.Conv2d):
                prune.random_structured(module, 'weight', amount=0.8, dim=0)
                prune.remove(module, 'weight')

        x = torch.randn((100, 3, 224, 224))
        y_src = model(x)

        simplify.propagate_bias(model, torch.zeros((1, 3, 224, 224)), [])
        y_prop = model(x)

        with self.subTest(test='propagate_bias'):
            self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))

        simplify.remove_zeroed(model, torch.zeros((1, 3, 224, 224)), [])
        y_prop = model(x)

        with self.subTest(test='remove_zeroed'):
            self.assertTrue(torch.allclose(y_src, y_prop, atol=1e-6))
    
