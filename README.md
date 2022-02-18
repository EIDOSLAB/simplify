# Simplify

[comment]: <> ([![tests]&#40;https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml/badge.svg&#41;]&#40;https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml&#41;)

Simplification of pruned models for accelerated inference.

## Installation

Simplify can be installed using pip:

```bash
pip3 install torch-simplify
```

or if you want to run the latest version of the code, you can install from git:

```bash
git clone https://github.com/EIDOSlab/simplify
cd simplify
pip3 install -r requirements.txt
```

****

## Usage

### Main function

For most scenarios the main `simplify` function will suffice. This function returns the simplified model.

#### Arguments

The expected arguments are:

- `model (torch.nn.Module)`: Module to be simplified i.e. the PyTorch's model.
- `x (torch.Tensor)`: zero-tensor of shape `[1, C, N, M]`, same as the model usual input.
- `bn_folding (List)`: List of tuple (`nn.Conv2d`, `nn.BatchNorm2d`) to be fused. If None it tries to evaluate them
  given the model. Default `None`.
- `fuse_bn (bool)`: If True, fuse the conv-bn tuple.
- `pinned_out (List)`: List of `nn.Modules` which output needs to remain of the original shape (e.g. layers related to a
  residual connection with a sum operation).

#### Minimal working example

```python
import torch
from torchvision import models
from simplify import simplify

model = models.resnet18()

# Apply some pruning strategy or load a pruned checkpoint

dummy_input = torch.zeros(1, 3, 224, 224)  # Tensor shape is that of a standard input for the given model
simplified_model = simplify(model, dummy_input)
```

### Submodules

The `simplify` function is composed of three different submodules: `fuse`, `propagate` and `remove`. Each module can be
used independently as needed.

#### fuse

Fuses adjacent Conv (or Linear) and BatchNorm layers.

#### propagate

Propagates non-zero bias of pruned neurons to remaining neurons of the next layers.

#### remove

Removes zeroed neurons from the architecture.

****

## Citing

If you use this software for research or application purposes, please use the following citation:

```bibtex
@article{bragagnolo2021simplify,
  title = {Simplify: A Python library for optimizing pruned neural networks},
  journal = {SoftwareX},
  volume = {17},
  pages = {100907},
  year = {2022},
  issn = {2352-7110},
  doi = {https://doi.org/10.1016/j.softx.2021.100907},
  url = {https://www.sciencedirect.com/science/article/pii/S2352711021001576},
  author = {Andrea Bragagnolo and Carlo Alberto Barbano},
}
```
