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
Update timestamp 09/06/2021 19:12:17

Random structured pruning amount = 50.0%

| Architecture       | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|
| alexnet            | 0.2897s ± 0.0503  | 0.1185s ± 0.0026  |
| vgg11              | 3.1443s ± 0.1589  | 1.2348s ± 0.0394  |
| vgg11_bn           | 3.8797s ± 0.1467  | 1.2343s ± 0.0454  |
| vgg13              | 4.4773s ± 0.0932  | 1.9440s ± 0.1109  |
| vgg13_bn           | 6.0232s ± 0.2760  | 1.8734s ± 0.0811  |
| vgg16              | 5.5083s ± 0.1880  | 2.2702s ± 0.0811  |
| vgg16_bn           | 7.1850s ± 0.2267  | 2.2364s ± 0.0961  |
| vgg19              | 6.6639s ± 0.1938  | 2.5652s ± 0.0746  |
| vgg19_bn           | 8.3650s ± 0.2687  | 2.6303s ± 0.1530  |
| resnet18           | 1.1779s ± 0.0498  | 0.6446s ± 0.0100  |
| resnet34           | 2.0665s ± 0.1810  | 1.0954s ± 0.0935  |
| resnet50           | 4.2819s ± 0.1770  | 2.5654s ± 0.0893  |
| resnet101          | 6.6143s ± 0.1600  | 3.7886s ± 0.0852  |
| resnet152          | 9.4440s ± 0.2140  | 5.2877s ± 0.0616  |
| squeezenet1_0      | 1.0496s ± 0.0329  | 0.5848s ± 0.0065  |
| squeezenet1_1      | 0.5824s ± 0.0244  | 0.3341s ± 0.0037  |
| inception_v3       | 2.0929s ± 0.0678  | 0.7564s ± 0.0099  |
| googlenet          | 1.6310s ± 0.0754  | 0.6293s ± 0.0103  |
| shufflenet_v2_x0_5 | 0.4433s ± 0.0133  | 0.4466s ± 0.0216  |
| shufflenet_v2_x1_0 | 0.5693s ± 0.0112  | 0.5300s ± 0.0144  |
| shufflenet_v2_x1_5 | 0.7649s ± 0.0202  | 0.7265s ± 0.0208  |
| shufflenet_v2_x2_0 | 1.2785s ± 0.0286  | 1.0461s ± 0.0139  |
| mobilenet_v2       | 2.8491s ± 0.0573  | 2.5284s ± 0.0488  |
| mobilenet_v3_small | 0.7676s ± 0.0215  | 0.7478s ± 0.0285  |
| mobilenet_v3_large | 1.9852s ± 0.0462  | 1.7503s ± 0.0499  |
| resnext50_32x4d    | 5.1172s ± 0.1612  | 3.6673s ± 0.0595  |
| resnext101_32x8d   | 13.1092s ± 0.2916 | 8.8046s ± 0.1130  |
| wide_resnet50_2    | 6.8694s ± 0.2837  | 3.3443s ± 0.0764  |
| wide_resnet101_2   | 11.7889s ± 0.2877 | 4.8303s ± 0.0874  |
| mnasnet0_5         | 1.4428s ± 0.0438  | 1.3573s ± 0.0561  |
| mnasnet0_75        | 2.2419s ± 0.0547  | 2.1083s ± 0.0418  |
| mnasnet1_0         | 2.7525s ± 0.0431  | 2.4746s ± 0.0260  |
| mnasnet1_3         | 3.6940s ± 0.0754  | 3.2427s ± 0.0963  |
<!-- benchmark ends -->

### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 09/06/2021 17:00:55

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