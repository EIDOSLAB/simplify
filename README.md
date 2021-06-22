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
import torch
from torch import nn
from torch.nn.utils import prune
from torchvision.models import alexnet

from simplify import simplify

model = alexnet(pretrained=True)
model.eval()

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.random_structured(module, 'weight', amount=0.8, dim=0)
        prune.remove(module, 'weight')

zeros = torch.zeros(1, 3, 224, 224)
simplify(model, zeros)
```

<details>
<summary>
Benchmarks
</summary>


<!-- benchmark starts -->
Update timestamp 22/06/2021 21:39:39

Random structured pruning amount = 50.0%

| Architecture       | Pruned time      | Simplified time   |
|--------------------|------------------|-------------------|
| alexnet            | 0.2537s ± 0.0073 | 0.1096s ± 0.0024  |
| vgg11              | 2.7648s ± 0.0371 | 1.2164s ± 0.0038  |
| vgg11_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg13              | 4.1843s ± 0.0456 | 1.8703s ± 0.0030  |
| vgg13_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg16              | 5.1764s ± 0.0036 | 2.1973s ± 0.0008  |
| vgg16_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| vgg19              | 6.2071s ± 0.0298 | 2.5273s ± 0.0014  |
| vgg19_bn           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet18           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet34           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet50           | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet101          | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnet152          | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| squeezenet1_0      | 1.0406s ± 0.0020 | 1.1063s ± 0.0171  |
| squeezenet1_1      | 0.5779s ± 0.0029 | 0.6165s ± 0.0053  |
| densenet121        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet161        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet169        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| densenet201        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| inception_v3       | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| googlenet          | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x0_5 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x1_0 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x1_5 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| shufflenet_v2_x2_0 | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mobilenet_v2       | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mobilenet_v3_small | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mobilenet_v3_large | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnext50_32x4d    | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| resnext101_32x8d   | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| wide_resnet50_2    | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| wide_resnet101_2   | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet0_5         | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet0_75        | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet1_0         | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
| mnasnet1_3         | 0.0000s ± 0.0000 | 0.0000s ± 0.0000  |
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