#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
import unittest

import torch

from simplify import fuse
from simplify.utils import get_bn_folding
from test.utils import models, get_model


class Test(unittest.TestCase):
    def test(self):
        @torch.no_grad()
        def test_arch(arch, x):
            model = get_model(architecture, arch)

            y_src = model(x)
            bn_folding = get_bn_folding(model)
            model = fuse(model, bn_folding)
            y_prop = model(x)

            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))

        im = torch.randint(0, 256, (256, 3, 224, 224))
        x = im / 255.

        for architecture in models:
            print(f"Testing with {architecture.__name__}")
            with self.subTest(arch=architecture):
                self.assertTrue(test_arch(architecture, x))
