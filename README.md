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

## Usage

*Simplify* is composed of three modules: *fuse*, *propagate* and *remove*.

### Fuse

The *fuse* module implements standard BatchNorm fusion procedure. It requires a list of pairs (conv, bn) representing
the pairs of layers to fuse. This modules can be used in such order or independently from one another according to needs.

It can be used as:

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
Benchmarks
</summary>


<!-- benchmark starts -->
Update timestamp 23/06/2021 11:38:39

Random structured pruning amount = 50.0%

| Architecture       | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|
| alexnet            | 0.2567s ± 0.0147  | 0.1103s ± 0.0019  |
| vgg11              | 2.7595s ± 0.0277  | 1.2177s ± 0.0034  |
| vgg11_bn           | 3.6554s ± 0.0097  | 1.2040s ± 0.0023  |
| vgg13              | 4.1491s ± 0.0148  | 1.8530s ± 0.0083  |
| vgg13_bn           | 5.6327s ± 0.0094  | 1.8437s ± 0.0033  |
| vgg16              | 5.1179s ± 0.0050  | 2.1656s ± 0.0082  |
| vgg16_bn           | 6.7249s ± 0.0055  | 2.1635s ± 0.0032  |
| vgg19              | 6.1281s ± 0.0413  | 2.4808s ± 0.0014  |
| vgg19_bn           | 7.8951s ± 0.0256  | 2.5016s ± 0.0030  |
| resnet18           | 1.0492s ± 0.0130  | 0.6266s ± 0.0038  |
| resnet34           | 1.7541s ± 0.0578  | 0.9567s ± 0.0030  |
| resnet50           | 4.0148s ± 0.0330  | 2.5540s ± 0.0134  |
| resnet101          | 6.1164s ± 0.0050  | 3.7594s ± 0.0054  |
| resnet152          | 8.4297s ± 0.0113  | 5.1408s ± 0.0055  |
| squeezenet1_0      | 1.0064s ± 0.0019  | 1.0719s ± 0.0098  |
| squeezenet1_1      | 0.5562s ± 0.0015  | 0.5900s ± 0.0007  |
| densenet121        | 4.3583s ± 0.0173  | 4.0141s ± 0.1426  |
| densenet161        | 8.7723s ± 0.0365  | 7.7275s ± 0.0201  |
| densenet169        | 5.1136s ± 0.0351  | 4.6250s ± 0.0344  |
| densenet201        | 6.5296s ± 0.0257  | 6.1737s ± 0.0361  |
| inception_v3       | 1.9546s ± 0.0366  | 1.1694s ± 0.0070  |
| googlenet          | 1.4704s ± 0.0076  | 0.5809s ± 0.0086  |
| shufflenet_v2_x0_5 | 0.3820s ± 0.0027  | 0.3811s ± 0.0033  |
| shufflenet_v2_x1_0 | 0.5042s ± 0.0037  | 0.4826s ± 0.0012  |
| shufflenet_v2_x1_5 | 0.7095s ± 0.0033  | 0.6669s ± 0.0005  |
| shufflenet_v2_x2_0 | 1.1705s ± 0.0160  | 0.9611s ± 0.0024  |
| mobilenet_v2       | 2.5662s ± 0.0556  | 2.1873s ± 0.0545  |
| mobilenet_v3_small | 0.6592s ± 0.0101  | 0.6375s ± 0.0028  |
| mobilenet_v3_large | 1.7787s ± 0.0177  | 1.5868s ± 0.0157  |
| resnext50_32x4d    | 4.8473s ± 0.0014  | 3.7550s ± 0.0045  |
| resnext101_32x8d   | 12.1780s ± 0.0098 | 8.9929s ± 0.0152  |
| wide_resnet50_2    | 6.2961s ± 0.0107  | 3.1973s ± 0.0078  |
| wide_resnet101_2   | 10.4598s ± 0.0523 | 4.3079s ± 0.0454  |
| mnasnet0_5         | 1.2035s ± 0.0129  | 1.1411s ± 0.0033  |
| mnasnet0_75        | 1.9979s ± 0.0100  | 1.7071s ± 0.0166  |
| mnasnet1_0         | 2.3457s ± 0.0484  | 2.0913s ± 0.0035  |
| mnasnet1_3         | 3.3775s ± 0.0822  | 2.7992s ± 0.0630  |

<!-- benchmark ends -->

### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 22/06/2021 14:16:31

|    Architecture    |  BatchNorm Folding  |  Bias Propagation  |   Simplification   |
|--------------------|---------------------|--------------------|--------------------|
|      alexnet       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|       vgg11        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg11_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|       vgg13        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg13_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|       vgg16        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg16_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|       vgg19        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg19_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet18      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet34      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     resnet101      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     resnet152      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_0    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet161     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet169     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet201     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x0_5 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x1_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x1_5 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    mobilenet_v2    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| mobilenet_v3_small | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext50_32x4d   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet50_2   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet0_5     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    mnasnet0_75     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet1_3     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |

<!-- table ends -->
</details>