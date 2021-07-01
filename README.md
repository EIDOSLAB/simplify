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
Update timestamp 28/06/2021 17:51:58

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2551s ± 0.0066  | 0.2479s ± 0.0020  | 0.1090s ± 0.0024  |
| vgg11              | 2.8025s ± 0.0312  | 2.7569s ± 0.0125  | 1.2174s ± 0.0051  |
| vgg11_bn           | 3.6605s ± 0.0100  | 3.6314s ± 0.0018  | 1.1967s ± 0.0009  |
| vgg13              | 4.2164s ± 0.0191  | 4.1675s ± 0.0020  | 1.8718s ± 0.0121  |
| vgg13_bn           | 5.7609s ± 0.0104  | 5.7223s ± 0.0368  | 1.8758s ± 0.0018  |
| vgg16              | 5.2331s ± 0.0039  | 5.2154s ± 0.0039  | 2.2030s ± 0.0065  |
| vgg16_bn           | 6.9403s ± 0.0069  | 6.8955s ± 0.0314  | 2.2101s ± 0.0087  |
| vgg19              | 6.2741s ± 0.0498  | 6.2580s ± 0.0050  | 2.5279s ± 0.0106  |
| vgg19_bn           | 8.0622s ± 0.0730  | 8.0189s ± 0.0068  | 2.5049s ± 0.0015  |
| resnet18           | 1.0777s ± 0.0035  | 1.0601s ± 0.0009  | 0.6399s ± 0.0062  |
| resnet34           | 1.7961s ± 0.0088  | 1.7938s ± 0.0315  | 0.9900s ± 0.0007  |
| resnet50           | 4.0952s ± 0.0269  | 4.0866s ± 0.0320  | 2.5800s ± 0.0046  |
| resnet101          | 6.2304s ± 0.0169  | 6.4320s ± 0.4361  | 3.8157s ± 0.0019  |
| resnet152          | 8.7621s ± 0.0536  | 8.7335s ± 0.0043  | 5.3100s ± 0.0037  |
| squeezenet1_0      | 1.0788s ± 0.0025  | 1.0426s ± 0.0034  | 1.1002s ± 0.0106  |
| squeezenet1_1      | 0.6008s ± 0.0011  | 0.5803s ± 0.0005  | 0.6169s ± 0.0008  |
| densenet121        | 4.5417s ± 0.0262  | 4.5202s ± 0.0213  | 4.0234s ± 0.0247  |
| densenet161        | 8.7458s ± 0.0286  | 8.7196s ± 0.0184  | 7.5991s ± 0.0211  |
| densenet169        | 4.9435s ± 0.0678  | 4.9036s ± 0.0082  | 4.5785s ± 0.0112  |
| densenet201        | 6.3821s ± 0.0314  | 6.3550s ± 0.0102  | 6.0924s ± 0.0226  |
| inception_v3       | 1.9629s ± 0.0077  | 1.9340s ± 0.0101  | 1.1414s ± 0.0075  |
| googlenet          | 1.4318s ± 0.0370  | 1.3392s ± 0.0007  | 0.5483s ± 0.0061  |
| shufflenet_v2_x0_5 | 0.3830s ± 0.0005  | 0.3790s ± 0.0005  | 0.3675s ± 0.0032  |
| shufflenet_v2_x1_0 | 0.5022s ± 0.0033  | 0.4945s ± 0.0012  | 0.4755s ± 0.0009  |
| shufflenet_v2_x1_5 | 0.6860s ± 0.0052  | 0.6815s ± 0.0005  | 0.6530s ± 0.0011  |
| shufflenet_v2_x2_0 | 1.0187s ± 0.0054  | 1.0073s ± 0.0013  | 0.9210s ± 0.0009  |
| mobilenet_v2       | 2.4853s ± 0.0136  | 2.4390s ± 0.0447  | 2.1178s ± 0.0040  |
| mobilenet_v3_small | 0.6690s ± 0.0104  | 0.6764s ± 0.0063  | 0.6496s ± 0.0063  |
| mobilenet_v3_large | 1.7902s ± 0.0120  | 1.7308s ± 0.0367  | 1.5555s ± 0.0103  |
| resnext50_32x4d    | 4.5819s ± 0.0042  | 4.5648s ± 0.0015  | 3.5939s ± 0.0014  |
| resnext101_32x8d   | 12.1708s ± 0.0121 | 12.2176s ± 0.0285 | 9.0320s ± 0.0055  |
| wide_resnet50_2    | 6.4415s ± 0.0076  | 6.4234s ± 0.0121  | 3.1897s ± 0.0220  |
| wide_resnet101_2   | 10.1057s ± 0.0041 | 10.1886s ± 0.0135 | 4.5023s ± 0.0168  |
| mnasnet0_5         | 1.2945s ± 0.0033  | 1.2918s ± 0.0068  | 1.1949s ± 0.0049  |
| mnasnet0_75        | 2.0525s ± 0.0141  | 2.0401s ± 0.0169  | 1.7142s ± 0.0192  |
| mnasnet1_0         | 2.2915s ± 0.0038  | 2.3159s ± 0.0405  | 2.0594s ± 0.0022  |
| mnasnet1_3         | 3.2810s ± 0.0560  | 3.3705s ± 0.0670  | 2.7184s ± 0.0336  |

<!-- benchmark ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 01/07/2021 10:54:36

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
|      resnet18      | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|      resnet34      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     resnet101      | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|     resnet152      | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|   squeezenet1_0    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|    densenet161     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet169     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|    densenet201     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x0_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| shufflenet_v2_x1_0 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| shufflenet_v2_x1_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|    mobilenet_v2    | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| mobilenet_v3_small | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|  resnext50_32x4d   | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|  wide_resnet50_2   | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|     mnasnet0_5     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|    mnasnet0_75     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|     mnasnet1_3     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
<!-- table ends -->
</details>