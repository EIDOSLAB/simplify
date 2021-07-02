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
Update timestamp 02/07/2021 14:12:39

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2541s ± 0.0105  | 0.2457s ± 0.0033  | 0.1114s ± 0.0028  |
| vgg11              | 2.8115s ± 0.0270  | 2.7708s ± 0.0072  | 1.2147s ± 0.0082  |
| vgg11_bn           | 3.7360s ± 0.0390  | 3.7012s ± 0.0077  | 1.2138s ± 0.0043  |
| vgg13              | 4.2442s ± 0.0054  | 4.2092s ± 0.0106  | 1.8750s ± 0.0044  |
| vgg13_bn           | 5.7653s ± 0.0091  | 5.7342s ± 0.0106  | 1.8787s ± 0.0087  |
| vgg16              | 5.2710s ± 0.0143  | 5.2519s ± 0.0060  | 2.2150s ± 0.0176  |
| vgg16_bn           | 6.9713s ± 0.0282  | 6.8915s ± 0.0264  | 2.2289s ± 0.0724  |
| vgg19              | 6.3143s ± 0.0450  | 6.3027s ± 0.0038  | 2.5350s ± 0.0090  |
| vgg19_bn           | 8.0637s ± 0.0083  | 8.0679s ± 0.0191  | 2.5396s ± 0.0126  |
| resnet18           | 1.1364s ± 0.1062  | 1.0868s ± 0.0075  | 0.7186s ± 0.0030  |
| resnet34           | 1.8281s ± 0.0591  | 1.7880s ± 0.0102  | 1.0856s ± 0.0027  |
| resnet50           | 4.0952s ± 0.0078  | 4.1012s ± 0.0370  | 3.0019s ± 0.0042  |
| resnet101          | 6.2443s ± 0.0137  | 6.2832s ± 0.0760  | 4.5807s ± 0.0081  |
| resnet152          | 8.8325s ± 0.1453  | 8.8212s ± 0.0768  | 6.4051s ± 0.0053  |
| squeezenet1_0      | 1.0912s ± 0.0298  | 1.0505s ± 0.0034  | 1.2466s ± 0.0038  |
| squeezenet1_1      | 0.6031s ± 0.0030  | 0.5828s ± 0.0032  | 0.6849s ± 0.0041  |
| densenet121        | 4.5933s ± 0.0279  | 4.5877s ± 0.0441  | 4.9396s ± 0.0133  |
| densenet161        | 9.1743s ± 0.0391  | 9.1517s ± 0.0713  | 9.4493s ± 0.0485  |
| densenet169        | 5.2091s ± 0.0689  | 5.1649s ± 0.0149  | 5.8039s ± 0.0133  |
| densenet201        | 6.7510s ± 0.0166  | 6.7363s ± 0.0214  | 7.6527s ± 0.0303  |
| inception_v3       | 2.0370s ± 0.0086  | 1.9996s ± 0.0026  | 1.2830s ± 0.0073  |
| googlenet          | 1.5790s ± 0.0176  | 1.4870s ± 0.0097  | 0.5660s ± 0.0149  |
| shufflenet_v2_x0_5 | 0.3875s ± 0.0014  | 0.3806s ± 0.0037  | 0.4094s ± 0.0023  |
| shufflenet_v2_x1_0 | 0.4957s ± 0.0032  | 0.4915s ± 0.0019  | 0.5379s ± 0.0025  |
| shufflenet_v2_x1_5 | 0.7417s ± 0.0090  | 0.7353s ± 0.0033  | 0.7897s ± 0.0040  |
| shufflenet_v2_x2_0 | 1.1501s ± 0.0200  | 1.1464s ± 0.0044  | 1.1498s ± 0.0054  |
| mobilenet_v2       | 2.6127s ± 0.0453  | 2.6065s ± 0.0353  | 2.4583s ± 0.0307  |
| mobilenet_v3_small | 0.6765s ± 0.0040  | 0.6761s ± 0.0023  | 0.6969s ± 0.0100  |
| mobilenet_v3_large | 1.8596s ± 0.0428  | 1.8149s ± 0.0073  | 1.7527s ± 0.0074  |
| resnext50_32x4d    | 5.0396s ± 0.0337  | 5.0199s ± 0.0108  | 4.4090s ± 0.0100  |
| resnext101_32x8d   | 12.3411s ± 0.0514 | 12.3163s ± 0.1183 | 10.4645s ± 0.0116 |
| wide_resnet50_2    | 6.5117s ± 0.0405  | 6.4444s ± 0.0086  | 3.6526s ± 0.0063  |
| wide_resnet101_2   | 10.4351s ± 0.0752 | 10.8512s ± 0.0108 | 5.2682s ± 0.0236  |
| mnasnet0_5         | 1.2658s ± 0.0213  | 1.2897s ± 0.0219  | 1.2825s ± 0.0319  |
| mnasnet0_75        | 2.1257s ± 0.0086  | 2.1124s ± 0.0113  | 2.0052s ± 0.0417  |
| mnasnet1_0         | 2.3901s ± 0.0488  | 2.4365s ± 0.0391  | 2.3462s ± 0.0382  |
| mnasnet1_3         | 3.2735s ± 0.0650  | 3.2876s ± 0.0669  | 3.1677s ± 0.0657  |
<!-- benchmark ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 02/07/2021 13:00:59

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