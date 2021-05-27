import unittest
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from torchvision.models.alexnet import alexnet
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152

import utils
from utils import set_seed
from fuser import fuse
from simplify import __propagate_bias as propagate_bias
from simplify import __remove_zeroed as remove_zeroed

class ChannelsRemovalTest(unittest.TestCase):
    def setUp(self):
        set_seed(3)

    def test_zeroed_removal(self):

        @torch.no_grad()
        def test_arch(arch, x, pretrained=False):
            model = arch(pretrained, progress=False)
            model.eval()
            
            for module in model.modules():
                if isinstance(module, nn.Conv2d):
                    prune.random_structured(module, 'weight', amount=0.8, dim=0)
                    prune.remove(module, 'weight')
            
            pinned_out = utils.get_pinned_out(model)

            model = fuse(model)
            zeros = torch.zeros(1, *x.shape[1:])
            propagate_bias(model, zeros, pinned_out)
            y_src = model(x)
            
            model = remove_zeroed(model, pinned_out)    
            y_prop = model(x)
            
            print(f'------ {self.__class__.__name__, arch.__name__} ------')
            print("Max abs diff: ", (y_src - y_prop).abs().max().item())
            print("MSE diff: ", nn.MSELoss()(y_src, y_prop).item())
            print(f'Correct predictions: {torch.eq(y_src.argmax(dim=1), y_prop.argmax(dim=1)).sum()}/{y_prop.shape[0]}')
            print()
            
            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))
        
        im = torch.randint(0, 256, ((256, 3, 224, 224)))
        x = im / 255.
        
        for architecture in [alexnet, vgg16, vgg16_bn, resnet18, resnet34, resnet50, resnet101, resnet152]:
            with self.subTest(arch=architecture, pretrained=True):
                self.assertTrue(test_arch(architecture, x, True))
