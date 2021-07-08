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

<!-- benchmark starts -->
Update timestamp 08/07/2021 13:57:18

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg11              | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg11_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg13              | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg13_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg16              | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg16_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg19              | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg19_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet18           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet34           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet50           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet101          | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet152          | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| squeezenet1_0      | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| squeezenet1_1      | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet121        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet161        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet169        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet201        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| inception_v3       | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| googlenet          | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x0_5 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x1_0 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x1_5 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x2_0 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mobilenet_v2       | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mobilenet_v3_small | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mobilenet_v3_large | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnext50_32x4d    | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnext101_32x8d   | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| wide_resnet50_2    | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| wide_resnet101_2   | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet0_5         | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet0_75        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet1_0         | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet1_3         | 0.0000s ± 0.0000 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
<!-- benchmark ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 03/07/2021 15:10:26

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