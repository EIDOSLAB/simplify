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
Update timestamp 05/07/2021 18:17:31

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2577s ± 0.0081  | 0.2504s ± 0.0033  | 0.1111s ± 0.0020  |
| vgg11              | 2.8114s ± 0.0063  | 2.7723s ± 0.0049  | 1.2322s ± 0.0329  |
| vgg11_bn           | 3.7295s ± 0.0315  | 3.7032s ± 0.0359  | 1.2042s ± 0.0036  |
| vgg13              | 4.2336s ± 0.0087  | 4.1931s ± 0.0052  | 1.8775s ± 0.0155  |
| vgg13_bn           | 5.7881s ± 0.0093  | 5.7439s ± 0.0155  | 1.8816s ± 0.0096  |
| vgg16              | 5.2830s ± 0.0519  | 5.2388s ± 0.0097  | 2.2093s ± 0.0159  |
| vgg16_bn           | 7.0211s ± 0.0519  | 6.9299s ± 0.0098  | 2.2177s ± 0.0161  |
| vgg19              | 6.3392s ± 0.0596  | 6.3078s ± 0.0104  | 2.5403s ± 0.0049  |
| vgg19_bn           | 8.1515s ± 0.0105  | 8.1052s ± 0.0129  | 2.5446s ± 0.0122  |
| resnet18           | 1.1034s ± 0.0086  | 1.0807s ± 0.0392  | 0.7747s ± 0.0088  |
| resnet34           | 1.7694s ± 0.0198  | 1.7521s ± 0.0231  | 1.1362s ± 0.0044  |
| resnet50           | 4.1352s ± 0.0597  | 4.1138s ± 0.0125  | 3.2767s ± 0.0309  |
| resnet101          | 6.0592s ± 0.0131  | 6.0507s ± 0.0195  | 4.7266s ± 0.0113  |
| resnet152          | 8.5031s ± 0.1002  | 8.8667s ± 0.1515  | 6.7244s ± 0.0086  |
| squeezenet1_0      | 1.0848s ± 0.0133  | 1.0484s ± 0.0036  | 1.3925s ± 0.0031  |
| squeezenet1_1      | 0.6057s ± 0.0044  | 0.5827s ± 0.0031  | 0.7286s ± 0.0069  |
| densenet121        | 4.6411s ± 0.0149  | 4.6228s ± 0.0108  | 4.9788s ± 0.0547  |
| densenet161        | 8.7623s ± 0.1674  | 8.6944s ± 0.0281  | 9.1983s ± 0.0217  |
| densenet169        | 4.9993s ± 0.0833  | 4.9432s ± 0.0111  | 5.6277s ± 0.0145  |
| densenet201        | 6.4206s ± 0.0077  | 6.4032s ± 0.0125  | 7.3703s ± 0.0387  |
| inception_v3       | 1.9746s ± 0.0095  | 1.9572s ± 0.0306  | 1.2341s ± 0.0076  |
| googlenet          | 1.4476s ± 0.0392  | 1.3462s ± 0.0040  | 0.5527s ± 0.0059  |
| shufflenet_v2_x0_5 | 0.3829s ± 0.0015  | 0.3785s ± 0.0010  | 0.4135s ± 0.0004  |
| shufflenet_v2_x1_0 | 0.4860s ± 0.0053  | 0.4784s ± 0.0026  | 0.5126s ± 0.0016  |
| shufflenet_v2_x1_5 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| shufflenet_v2_x2_0 | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mobilenet_v2       | 2.4628s ± 0.0406  | 2.4392s ± 0.0059  | 2.4253s ± 0.0251  |
| mobilenet_v3_small | 0.6632s ± 0.0078  | 0.6768s ± 0.0014  | 0.6897s ± 0.0079  |
| mobilenet_v3_large | 1.6865s ± 0.0138  | 1.6973s ± 0.0323  | 1.7647s ± 0.0304  |
| resnext50_32x4d    | 4.6971s ± 0.0084  | 4.6811s ± 0.0144  | 4.6095s ± 0.0200  |
| resnext101_32x8d   | 11.8239s ± 0.0175 | 11.8258s ± 0.1137 | 10.4550s ± 0.0122 |
| wide_resnet50_2    | 6.1868s ± 0.0124  | 6.1684s ± 0.0092  | 3.7727s ± 0.0111  |
| wide_resnet101_2   | 10.1057s ± 0.1189 | 10.0348s ± 0.0206 | 5.0008s ± 0.0687  |
| mnasnet0_5         | 1.1340s ± 0.0117  | 1.1301s ± 0.0051  | 1.1806s ± 0.0179  |
| mnasnet0_75        | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  | 0.0000s ± 0.0000  |
| mnasnet1_0         | 2.2791s ± 0.0163  | 2.3859s ± 0.0118  | 2.3062s ± 0.0378  |
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