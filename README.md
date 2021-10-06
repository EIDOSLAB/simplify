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
Update timestamp 05/10/2021 12:06:08

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0231s ± 0.0138 | 0.0179s ± 0.0058 | 0.0165s ± 0.0093  |
| densenet121        | 0.1791s ± 0.0030 | 0.1859s ± 0.0103 | 0.1182s ± 0.0086  |
| googlenet          | 0.1070s ± 0.0069 | 0.0943s ± 0.0078 | 0.0473s ± 0.0018  |
| inception_v3       | 0.1550s ± 0.0052 | 0.1405s ± 0.0062 | 0.0995s ± 0.0161  |
| mnasnet1_0         | 0.2155s ± 0.0025 | 0.2147s ± 0.0027 | 0.2251s ± 0.0216  |
| mobilenet_v3_large | 0.1261s ± 0.0014 | 0.1277s ± 0.0019 | 0.1211s ± 0.0021  |
| resnet50           | 0.2046s ± 0.0041 | 0.2111s ± 0.0074 | 0.1087s ± 0.0128  |
| resnext101_32x8d   | 1.4459s ± 0.1557 | 1.3544s ± 0.0512 | 0.7514s ± 0.1190  |
| shufflenet_v2_x2_0 | 0.0872s ± 0.0044 | 0.0791s ± 0.0036 | 0.0522s ± 0.0038  |
| squeezenet1_1      | 0.0392s ± 0.0019 | 0.0370s ± 0.0007 | 0.0284s ± 0.0010  |
| vgg19_bn           | 0.1194s ± 0.0079 | 0.1139s ± 0.0135 | 0.0552s ± 0.0059  |
| wide_resnet101_2   | 0.8139s ± 0.0444 | 0.7770s ± 0.0455 | 0.2899s ± 0.0425  |
<!-- benchmark eval ends -->

Training mode (leaves BatchNorm)

<!-- benchmark train starts -->
Update timestamp 05/10/2021 12:06:08

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0231s ± 0.0138 | 0.0179s ± 0.0058 | 0.0165s ± 0.0093  |
| densenet121        | 0.1791s ± 0.0030 | 0.1859s ± 0.0103 | 0.1182s ± 0.0086  |
| googlenet          | 0.1070s ± 0.0069 | 0.0943s ± 0.0078 | 0.0473s ± 0.0018  |
| inception_v3       | 0.1550s ± 0.0052 | 0.1405s ± 0.0062 | 0.0995s ± 0.0161  |
| mnasnet1_0         | 0.2155s ± 0.0025 | 0.2147s ± 0.0027 | 0.2251s ± 0.0216  |
| mobilenet_v3_large | 0.1261s ± 0.0014 | 0.1277s ± 0.0019 | 0.1211s ± 0.0021  |
| resnet50           | 0.2046s ± 0.0041 | 0.2111s ± 0.0074 | 0.1087s ± 0.0128  |
| resnext101_32x8d   | 1.4459s ± 0.1557 | 1.3544s ± 0.0512 | 0.7514s ± 0.1190  |
| shufflenet_v2_x2_0 | 0.0872s ± 0.0044 | 0.0791s ± 0.0036 | 0.0522s ± 0.0038  |
| squeezenet1_1      | 0.0392s ± 0.0019 | 0.0370s ± 0.0007 | 0.0284s ± 0.0010  |
| vgg19_bn           | 0.1194s ± 0.0079 | 0.1139s ± 0.0135 | 0.0552s ± 0.0059  |
| wide_resnet101_2   | 0.8139s ± 0.0444 | 0.7770s ± 0.0455 | 0.2899s ± 0.0425  |
<!-- benchmark train ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 05/10/2021 11:38:19

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
<!-- table ends -->
</details>