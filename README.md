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
Update timestamp 22/06/2021 22:31:16

Random structured pruning amount = 50.0%

| Architecture       | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|
| alexnet            | 0.2517s ± 0.0072  | 0.1101s ± 0.0019  |
| vgg11              | 2.7506s ± 0.0029  | 1.2190s ± 0.0042  |
| vgg11_bn           | 3.6760s ± 0.0025  | 1.2037s ± 0.0007  |
| vgg13              | 4.1692s ± 0.0066  | 1.8721s ± 0.0041  |
| vgg13_bn           | 5.6958s ± 0.0032  | 1.8769s ± 0.0060  |
| vgg16              | 5.1861s ± 0.0177  | 2.2600s ± 0.1813  |
| vgg16_bn           | 6.8564s ± 0.0178  | 2.1986s ± 0.0251  |
| vgg19              | 6.2218s ± 0.0308  | 2.5354s ± 0.0154  |
| vgg19_bn           | 8.0368s ± 0.0276  | 2.5269s ± 0.0041  |
| resnet18           | 1.0706s ± 0.0091  | 0.6369s ± 0.0024  |
| resnet34           | 1.7831s ± 0.0133  | 0.9845s ± 0.0005  |
| resnet50           | 4.0924s ± 0.0067  | 2.5814s ± 0.0025  |
| resnet101          | 6.2378s ± 0.0236  | 3.8173s ± 0.0038  |
| resnet152          | 8.7698s ± 0.0064  | 5.3222s ± 0.0053  |
| squeezenet1_0      | 1.0550s ± 0.0160  | 1.1031s ± 0.0056  |
| squeezenet1_1      | 0.5689s ± 0.0009  | 0.6070s ± 0.0004  |
| densenet121        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| densenet161        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| densenet169        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| densenet201        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| inception_v3       | 1.9843s ± 0.0103  | 1.2022s ± 0.0058  |
| googlenet          | 1.5601s ± 0.0107  | 0.6023s ± 0.0081  |
| shufflenet_v2_x0_5 | 0.3988s ± 0.0011  | 0.3957s ± 0.0013  |
| shufflenet_v2_x1_0 | 0.5176s ± 0.0025  | 0.5002s ± 0.0066  |
| shufflenet_v2_x1_5 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| shufflenet_v2_x2_0 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mobilenet_v2       | 2.6152s ± 0.0192  | 2.1978s ± 0.0022  |
| mobilenet_v3_small | 0.6817s ± 0.0030  | 0.6544s ± 0.0041  |
| mobilenet_v3_large | 1.8743s ± 0.0372  | 1.6697s ± 0.0063  |
| resnext50_32x4d    | 4.8441s ± 0.0035  | 3.7090s ± 0.0125  |
| resnext101_32x8d   | 11.8774s ± 0.0402 | 8.7001s ± 0.0105  |
| wide_resnet50_2    | 6.2758s ± 0.0105  | 3.1879s ± 0.0088  |
| wide_resnet101_2   | 10.4744s ± 0.0292 | 4.3912s ± 0.0739  |
| mnasnet0_5         | 1.2369s ± 0.0083  | 1.1703s ± 0.0159  |
| mnasnet0_75        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mnasnet1_0         | 2.4191s ± 0.0115  | 2.1094s ± 0.0091  |
| mnasnet1_3         | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
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