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
Update timestamp 05/07/2021 17:07:18

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2570s ± 0.0080  | 0.2506s ± 0.0030  | 0.1121s ± 0.0033  |
| vgg11              | 2.8136s ± 0.0275  | 2.7678s ± 0.0056  | 1.2171s ± 0.0050  |
| vgg11_bn           | 3.7149s ± 0.0078  | 3.6866s ± 0.0123  | 1.2011s ± 0.0038  |
| vgg13              | 4.2343s ± 0.0101  | 4.1918s ± 0.0075  | 1.8717s ± 0.0095  |
| vgg13_bn           | 5.7986s ± 0.0128  | 5.7409s ± 0.0275  | 1.8719s ± 0.0050  |
| vgg16              | 5.2692s ± 0.0390  | 5.2358s ± 0.0058  | 2.2099s ± 0.0069  |
| vgg16_bn           | 6.9908s ± 0.0110  | 6.9438s ± 0.0173  | 2.2302s ± 0.0282  |
| vgg19              | 6.3236s ± 0.0519  | 6.3148s ± 0.0317  | 2.5386s ± 0.0137  |
| vgg19_bn           | 8.1331s ± 0.0307  | 8.0853s ± 0.0147  | 2.5252s ± 0.0047  |
| resnet18           | 1.0944s ± 0.0126  | 1.0801s ± 0.0117  | 0.6377s ± 0.0033  |
| resnet34           | 1.8472s ± 0.0959  | 1.7973s ± 0.0056  | 0.9816s ± 0.0063  |
| resnet50           | 4.1063s ± 0.0316  | 4.0863s ± 0.0281  | 2.6312s ± 0.0031  |
| resnet101          | 6.2296s ± 0.0225  | 6.1892s ± 0.1236  | 4.0449s ± 0.0214  |
| resnet152          | 8.7160s ± 0.0425  | 8.7763s ± 0.0362  | 5.6640s ± 0.0069  |
| squeezenet1_0      | 1.0790s ± 0.0057  | 1.0473s ± 0.0076  | 1.1050s ± 0.0106  |
| squeezenet1_1      | 0.6026s ± 0.0036  | 0.5809s ± 0.0031  | 0.6106s ± 0.0027  |
| densenet121        | 4.6236s ± 0.0284  | 4.6028s ± 0.0345  | 4.5979s ± 0.0392  |
| densenet161        | 9.2074s ± 0.1030  | 9.1452s ± 0.0269  | 8.5068s ± 0.1067  |
| densenet169        | 5.0266s ± 0.0175  | 5.0091s ± 0.0083  | 5.1961s ± 0.0118  |
| densenet201        | 6.5205s ± 0.0159  | 6.5346s ± 0.0179  | 7.0007s ± 0.0187  |
| inception_v3       | 1.9880s ± 0.0132  | 1.9572s ± 0.0073  | 1.2013s ± 0.0087  |
| googlenet          | 1.4079s ± 0.0457  | 1.3147s ± 0.0042  | 0.5510s ± 0.0045  |
| shufflenet_v2_x0_5 | 0.3876s ± 0.0011  | 0.3837s ± 0.0011  | 0.3755s ± 0.0017  |
| shufflenet_v2_x1_0 | 0.4981s ± 0.0047  | 0.4915s ± 0.0029  | 0.4820s ± 0.0036  |
| shufflenet_v2_x1_5 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| shufflenet_v2_x2_0 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mobilenet_v2       | 2.4940s ± 0.0525  | 2.5458s ± 0.0242  | 2.2559s ± 0.0137  |
| mobilenet_v3_small | 0.6785s ± 0.0036  | 0.6771s ± 0.0016  | 0.6499s ± 0.0100  |
| mobilenet_v3_large | 1.8464s ± 0.0250  | 1.7853s ± 0.0300  | 1.6400s ± 0.0193  |
| resnext50_32x4d    | 4.8708s ± 0.0097  | 4.8742s ± 0.0259  | 3.7723s ± 0.0121  |
| resnext101_32x8d   | 11.8962s ± 0.1074 | 11.8011s ± 0.0315 | 8.9820s ± 0.0077  |
| wide_resnet50_2    | 6.3384s ± 0.0206  | 6.3067s ± 0.0115  | 3.1987s ± 0.0127  |
| wide_resnet101_2   | 10.6289s ± 0.0568 | 10.6243s ± 0.0145 | 4.6833s ± 0.0235  |
| mnasnet0_5         | 1.2356s ± 0.0269  | 1.2523s ± 0.0162  | 1.1601s ± 0.0144  |
| mnasnet0_75        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mnasnet1_0         | 2.3708s ± 0.0073  | 2.3779s ± 0.0127  | 2.1466s ± 0.0523  |
| mnasnet1_3         | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
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