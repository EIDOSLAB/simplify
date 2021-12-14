# Simplify

[![tests](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml/badge.svg)](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml)

Simplification of pruned models for accelerated inference.

[comment]: <> (- [Installation]&#40;#installation&#41;)

[comment]: <> (- [Modules]&#40;#usage&#41;)

[comment]: <> (    - [Dataloaders]&#40;#dataloaders&#41;)

[comment]: <> (    - [Evaluation]&#40;#evalutation&#41;)

[comment]: <> (    - [Models]&#40;#models&#41;)

[comment]: <> (    - [Pruning]&#40;#pruning&#41;)

[comment]: <> (        - [CSNN]&#40;#CSNN&#41;)

[comment]: <> (        - [Pruning]&#40;#Pruning&#41;)

[comment]: <> (        - [Thresholding]&#40;#Thresholding&#41;)

[comment]: <> (    - [Utils]&#40;#Utils&#41;)

[comment]: <> (- [Contributing]&#40;#contributing&#41;   )

[comment]: <> (- [License]&#40;#license&#41;)

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

## Example usage

```python
from torchvision.models import resnet18
from simplify import fuse

model = resnet18()
model.eval()
bn_folding = ...  # List of pairs (conv, bn) to fuse in a single layer
model = fuse(model, bn_folding)
```

### Propagate

The *propagate* module is used to remove the non-zero bias from zeroed-out neurons in order to be able to remove them.

````python
import torch
from simplify import propagate_bias
from torchvision.models import resnet18

zeros = torch.zeros(1, 3, 224, 224)
model = resnet18()
pinned_out = ...  # List of layers for which the bias should not be propagated
propagate_bias(model, zeros, pinned_out)
````

### Remove

The *remove* module is used to remove actually remove the zeroed neurons from the model architecture.

````python
import torch
from simplify import remove_zeroed
from torchvision.models import resnet18

zeros = torch.zeros(1, 3, 224, 224)
model = resnet18()
pinned_out = ...  # List of layers in which the output should not change shape
remove_zeroed(model, zeros, pinned_out)
````

### Utilities

We also provide a set of utilities used to define `bn_folding` and `pinned_out` for standard PyTorch models.

````python
from torchvision.models import resnet18
from utils import get_bn_folding, get_pinned_out

model = resnet18()
bn_folding = get_bn_folding(model)
pinned_out = get_pinned_out(model)
````

<details>
<summary>
Tests
</summary>

#### Inference time benchmarks

###### Evaluation mode (fuses BatchNorm)

<!-- benchmark eval starts -->
Update timestamp 08/10/2021 14:26:25

Random structured pruning amount = 50.0%

| Architecture       | Dense time      | Pruned time     | Simplified time   |
|--------------------|-----------------|-----------------|-------------------|
| alexnet            | 7.58ms ± 0.29   | 7.55ms ± 0.28   | 2.95ms ± 0.02     |
| densenet121        | 36.41ms ± 4.88  | 34.31ms ± 3.85  | 21.87ms ± 1.45    |
| googlenet          | 15.44ms ± 3.19  | 13.68ms ± 0.09  | 10.31ms ± 0.82    |
| inception_v3       | 25.29ms ± 7.31  | 21.68ms ± 2.90  | 13.22ms ± 2.23    |
| mnasnet1_0         | 17.66ms ± 0.57  | 13.64ms ± 0.13  | 11.59ms ± 0.07    |
| mobilenet_v3_large | 13.74ms ± 0.67  | 12.18ms ± 0.46  | 11.95ms ± 0.21    |
| resnet50           | 24.39ms ± 4.48  | 26.19ms ± 5.84  | 18.21ms ± 1.98    |
| resnext101_32x8d   | 76.11ms ± 15.79 | 77.35ms ± 20.04 | 65.68ms ± 16.41   |
| shufflenet_v2_x2_0 | 18.07ms ± 2.23  | 14.32ms ± 0.21  | 13.06ms ± 0.08    |
| squeezenet1_1      | 4.50ms ± 0.06   | 4.39ms ± 0.05   | 4.09ms ± 0.50     |
| vgg19_bn           | 40.41ms ± 12.13 | 38.56ms ± 10.72 | 12.39ms ± 0.19    |
| wide_resnet101_2   | 79.40ms ± 25.57 | 82.86ms ± 22.47 | 60.16ms ± 10.77   |
<!-- benchmark eval ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one

Fuse BatchNorm

<!-- table fuse starts -->
Update timestamp 06/10/2021 20:26:15

|    Architecture    |  BatchNorm Folding  |  Bias Propagation  |   Simplification   |
|--------------------|---------------------|--------------------|--------------------|
|      alexnet       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg19_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |

<!-- table fuse ends -->

Keep BatchNorm

<!-- table no fuse starts -->
Update timestamp 06/10/2021 20:36:11

|    Architecture    |  BatchNorm Folding  |  Bias Propagation  |   Simplification   |
|--------------------|---------------------|--------------------|--------------------|
|      alexnet       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg19_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |

<!-- table no fuse ends -->
</details>


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
