#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
import unittest
from copy import deepcopy

import torch

from simplify import simplify
from test.utils import models, get_model


class Test(unittest.TestCase):
    def test(self):
        @torch.no_grad()
        def test_arch(arch, x, fuse_bn):
            print(f"Fuse: {fuse_bn}")

            model = get_model(architecture, arch)
            y_src = model(x)

            zeros = torch.zeros(1, *x.shape[1:])
            model = simplify(deepcopy(model), zeros, fuse_bn=fuse_bn)
            y_simplified = model(x)

            self.assertTrue(torch.allclose(y_simplified, y_src, atol=1e-4)
                            & torch.equal(y_simplified.argmax(dim=1), y_src.argmax(dim=1)))

        im = torch.randint(0, 256, (16, 3, 224, 224))
        x = im / 255.

        for architecture in models:
            print(f"Testing with {architecture.__name__}")

            for i in range(20):
                with self.subTest(arch=architecture, fuse_bn=True):
                    test_arch(architecture, x, fuse_bn=True)

                with self.subTest(arch=architecture, fuse_bn=False):
                    test_arch(architecture, x, fuse_bn=False)
