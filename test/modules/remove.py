#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
import unittest

import torch

from simplify import fuse, propagate_bias, remove_zeroed
from simplify.utils import get_bn_folding, get_pinned
from test.utils import models, get_model


class Test(unittest.TestCase):
    def test(self):
        @torch.no_grad()
        def test_arch(arch, x, fuse_bn):
            print(f"Fuse: {fuse_bn}")

            model = get_model(architecture, arch)

            if fuse_bn:
                bn_folding = get_bn_folding(model)
                model = fuse(model, bn_folding)

            pinned_out = get_pinned(model)
            zeros = torch.zeros(1, *x.shape[1:])
            propagate_bias(model, zeros, pinned_out)
            y_src = model(x)

            model = remove_zeroed(model, zeros, pinned_out)
            y_prop = model(x)

            return torch.equal(y_src.argmax(dim=1), y_prop.argmax(dim=1))

        im = torch.randint(0, 256, (256, 3, 224, 224))
        x = im / 255.

        for architecture in models:
            print(f"Testing with {architecture.__name__}")

            with self.subTest(arch=architecture, fuse_bn=True):
                self.assertTrue(test_arch(architecture, x, fuse_bn=True))

            with self.subTest(arch=architecture, fuse_bn=False):
                self.assertTrue(test_arch(architecture, x, fuse_bn=False))
