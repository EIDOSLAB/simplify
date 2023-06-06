#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
import unittest
from copy import deepcopy

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
            model = fuse(deepcopy(model), bn_folding)
            y_fuse = model(x)

            self.assertTrue(torch.allclose(y_fuse, y_src, atol=1e-4)
                            & torch.equal(y_fuse.argmax(dim=1), y_src.argmax(dim=1)))

        im = torch.randint(0, 256, (16, 3, 224, 224))
        x = im / 255.

        for architecture in models:
            print(f"Testing with {architecture.__name__}")

            for i in range(50):
                with self.subTest(arch=architecture):
                    test_arch(architecture, x)
