import unittest

import simplify
import torchvision
from natsort import natsorted

class PinnedTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_pinned(self):
        def pinned_subtest(architecture):
            model = architecture()
            pinned = simplify.utils.get_pinned(model)
            print(f'{architecture.__name__}:', natsorted(pinned), '\n')
        
        pinned_subtest(torchvision.models.alexnet)
        pinned_subtest(torchvision.models.resnet18)
        pinned_subtest(torchvision.models.resnext101_32x8d)
        pinned_subtest(torchvision.models.mnasnet1_0)
        pinned_subtest(torchvision.models.densenet121)
        pinned_subtest(torchvision.models.squeezenet1_0)