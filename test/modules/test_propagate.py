#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).
import unittest
from copy import deepcopy

import torch
from torch._C._onnx import TrainingMode

from simplify import fuse, propagate_bias
from simplify.utils import get_bn_folding, get_pinned
from test.utils import models, get_model


class Test(unittest.TestCase):
    def test(self):
        @torch.no_grad()
        def test_arch(arch, x, fuse_bn):
            print(f"Fuse: {fuse_bn}")

            model = get_model(architecture, arch)
            y_src = model(x)

            if fuse_bn:
                bn_folding = get_bn_folding(model)
                model = fuse(deepcopy(model), bn_folding)
            y_fuse = model(x)

            self.assertTrue(torch.allclose(y_src, y_fuse, atol=1e-4)
                            & torch.equal(y_src.argmax(dim=1), y_fuse.argmax(dim=1)))

            zeros = torch.zeros(1, *x.shape[1:])
            pinned_out = get_pinned(model)
            model = propagate_bias(deepcopy(model), zeros, pinned_out)
            y_prop = model(x)

            self.assertTrue(torch.allclose(y_fuse, y_prop, atol=1e-4)
                            & torch.equal(y_fuse.argmax(dim=1), y_prop.argmax(dim=1)))

        im = torch.randint(0, 256, (1, 3, 224, 224))
        x = im / 255.

        for architecture in models:
            print(f"Testing with {architecture.__name__}")

            for i in range(100):
                with self.subTest(arch=architecture, fuse_bn=True):
                    test_arch(architecture, x, fuse_bn=True)

                with self.subTest(arch=architecture, fuse_bn=False):
                    test_arch(architecture, x, fuse_bn=False)
