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
Update timestamp 23/06/2021 10:20:42

Random structured pruning amount = 50.0%

| Architecture       | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|
| alexnet            | 0.2532s ± 0.0072  | 0.1093s ± 0.0028  |
| vgg11              | 2.7731s ± 0.0068  | 1.2109s ± 0.0024  |
| vgg11_bn           | 3.6989s ± 0.0057  | 1.2006s ± 0.0015  |
| vgg13              | 4.2101s ± 0.0037  | 1.8674s ± 0.0014  |
| vgg13_bn           | 5.7591s ± 0.0048  | 1.8795s ± 0.0059  |
| vgg16              | 5.2327s ± 0.0032  | 2.1981s ± 0.0038  |
| vgg16_bn           | 6.8722s ± 0.0039  | 2.2000s ± 0.0053  |
| vgg19              | 6.2533s ± 0.0297  | 2.5195s ± 0.0020  |
| vgg19_bn           | 8.0803s ± 0.0036  | 2.4986s ± 0.0033  |
| resnet18           | 1.0906s ± 0.0077  | 0.6380s ± 0.0042  |
| resnet34           | 1.7768s ± 0.0139  | 0.9863s ± 0.0059  |
| resnet50           | 4.0779s ± 0.0321  | 2.5833s ± 0.0051  |
| resnet101          | 6.2155s ± 0.0712  | 3.8184s ± 0.0030  |
| resnet152          | 8.8399s ± 0.0083  | 5.3276s ± 0.0056  |
| squeezenet1_0      | 1.0466s ± 0.0015  | 1.0997s ± 0.0074  |
| squeezenet1_1      | 0.5800s ± 0.0010  | 0.6162s ± 0.0008  |
| densenet121        | 4.5507s ± 0.0334  | 4.1944s ± 0.0118  |
| densenet161        | 9.1977s ± 0.1290  | 8.1085s ± 0.0143  |
| densenet169        | 5.2009s ± 0.0442  | 4.8804s ± 0.0071  |
| densenet201        | 6.7643s ± 0.0426  | 6.5154s ± 0.0253  |
| inception_v3       | 1.9953s ± 0.0075  | 1.1696s ± 0.0137  |
| googlenet          | 1.4191s ± 0.0377  | 0.5541s ± 0.0085  |
| shufflenet_v2_x0_5 | 0.3768s ± 0.0009  | 0.3678s ± 0.0010  |
| shufflenet_v2_x1_0 | 0.4882s ± 0.0019  | 0.4741s ± 0.0008  |
| shufflenet_v2_x1_5 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| shufflenet_v2_x2_0 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mobilenet_v2       | 2.5890s ± 0.0537  | 2.1957s ± 0.0338  |
| mobilenet_v3_small | 0.6412s ± 0.0132  | 0.6426s ± 0.0075  |
| mobilenet_v3_large | 1.8005s ± 0.0274  | 1.6181s ± 0.0198  |
| resnext50_32x4d    | 4.9589s ± 0.0044  | 3.8335s ± 0.0050  |
| resnext101_32x8d   | 12.4551s ± 0.0201 | 9.1058s ± 0.0095  |
| wide_resnet50_2    | 6.5808s ± 0.0039  | 3.2840s ± 0.0078  |
| wide_resnet101_2   | 10.9175s ± 0.0114 | 4.6249s ± 0.0157  |
| mnasnet0_5         | 1.1749s ± 0.0205  | 1.1392s ± 0.0182  |
| mnasnet0_75        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mnasnet1_0         | 2.4083s ± 0.0227  | 2.1678s ± 0.0170  |
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