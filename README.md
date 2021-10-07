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

Evaluation mode (fuses BatchNorm)

<!-- benchmark eval starts -->
Update timestamp 07/10/2021 09:28:41

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time               | Simplified time          |
|--------------------|------------------|---------------------------|--------------------------|
| alexnet            | 0.0158s ± 0.0043 | 0.0178s ± 0.0028 (112.58) | 0.0136s ± 0.0113 (85.98) |
| densenet121        | 0.0714s ± 0.0098 | 0.0765s ± 0.0054 (107.14) | 0.0563s ± 0.0034 (78.79) |
| googlenet          | 0.0275s ± 0.0012 | 0.0262s ± 0.0006 (95.16)  | 0.0189s ± 0.0048 (68.72) |
| inception_v3       | 0.0530s ± 0.0019 | 0.0534s ± 0.0021 (100.68) | 0.0267s ± 0.0012 (50.37) |
| mnasnet1_0         | 0.0335s ± 0.0022 | 0.0312s ± 0.0023 (93.09)  | 0.0313s ± 0.0018 (93.33) |
| mobilenet_v3_large | 0.0305s ± 0.0016 | 0.0299s ± 0.0017 (97.97)  | 0.0294s ± 0.0022 (96.36) |
| resnet50           | 0.0571s ± 0.0014 | 0.0624s ± 0.0139 (109.30) | 0.0394s ± 0.0032 (69.05) |
| resnext101_32x8d   | 0.1907s ± 0.0413 | 0.1944s ± 0.0238 (101.94) | 0.1656s ± 0.0338 (86.83) |
| shufflenet_v2_x2_0 | 0.0435s ± 0.0057 | 0.0390s ± 0.0027 (89.52)  | 0.0301s ± 0.0017 (69.17) |
| squeezenet1_1      | 0.0114s ± 0.0015 | 0.0122s ± 0.0016 (106.44) | 0.0090s ± 0.0010 (79.04) |
| vgg19_bn           | 0.0850s ± 0.0031 | 0.0854s ± 0.0034 (100.46) | 0.0353s ± 0.0007 (41.48) |
| wide_resnet101_2   | 0.1622s ± 0.0037 | 0.1522s ± 0.0010 (93.85)  | 0.0984s ± 0.0049 (60.69) |
<!-- benchmark eval ends -->

Training mode (keeps BatchNorm)

<!-- benchmark train starts -->
Update timestamp 06/10/2021 09:17:52

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0229s ± 0.0140 | 0.0189s ± 0.0066 | 0.0175s ± 0.0017  |
| densenet121        | 0.1849s ± 0.0115 | 0.2315s ± 0.0104 | 0.2098s ± 0.0087  |
| googlenet          | 0.1008s ± 0.0011 | 0.1014s ± 0.0011 | 0.0465s ± 0.0005  |
| inception_v3       | 0.1408s ± 0.0047 | 0.1606s ± 0.0028 | 0.1197s ± 0.0053  |
| mnasnet1_0         | 0.2214s ± 0.0018 | 0.2419s ± 0.0013 | 0.2214s ± 0.0125  |
| mobilenet_v3_large | 0.1352s ± 0.0017 | 0.1450s ± 0.0012 | 0.1418s ± 0.0063  |
| resnet50           | 0.2187s ± 0.0025 | 0.2217s ± 0.0079 | 0.1223s ± 0.0071  |
| resnext101_32x8d   | 1.4764s ± 0.1845 | 1.4690s ± 0.0527 | 1.3259s ± 0.0517  |
| shufflenet_v2_x2_0 | 0.0786s ± 0.0025 | 0.1038s ± 0.0028 | 0.0744s ± 0.0025  |
| squeezenet1_1      | 0.0362s ± 0.0013 | 0.0350s ± 0.0003 | 0.0356s ± 0.0012  |
| vgg19_bn           | 0.1137s ± 0.0022 | 0.1004s ± 0.0017 | 0.0469s ± 0.0021  |
| wide_resnet101_2   | 0.7587s ± 0.0107 | 0.8054s ± 0.0613 | 0.3695s ± 0.0168  |

<!-- benchmark train ends -->

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