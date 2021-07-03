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
Update timestamp 03/07/2021 14:18:42

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2567s ± 0.0098  | 0.2508s ± 0.0043  | 0.1116s ± 0.0031  |
| vgg11              | 2.7993s ± 0.0069  | 2.7697s ± 0.0101  | 1.2229s ± 0.0052  |
| vgg11_bn           | 3.7049s ± 0.0098  | 3.6924s ± 0.0528  | 1.2185s ± 0.0092  |
| vgg13              | 4.2388s ± 0.0045  | 4.2040s ± 0.0335  | 1.8826s ± 0.0201  |
| vgg13_bn           | 5.7982s ± 0.0339  | 5.7390s ± 0.0125  | 1.8806s ± 0.0052  |
| vgg16              | 5.2554s ± 0.0090  | 5.2340s ± 0.0089  | 2.2184s ± 0.0032  |
| vgg16_bn           | 7.0065s ± 0.0914  | 6.9189s ± 0.0106  | 2.2261s ± 0.0233  |
| vgg19              | 6.3209s ± 0.0407  | 6.3016s ± 0.0254  | 2.5461s ± 0.0080  |
| vgg19_bn           | 8.1354s ± 0.0281  | 8.0948s ± 0.0206  | 2.5457s ± 0.0301  |
| resnet18           | 1.0748s ± 0.0127  | 1.0551s ± 0.0090  | 0.6171s ± 0.0042  |
| resnet34           | 1.7778s ± 0.0372  | 1.7421s ± 0.0195  | 0.9525s ± 0.0039  |
| resnet50           | 4.1203s ± 0.0223  | 4.0793s ± 0.0570  | 2.6477s ± 0.0092  |
| resnet101          | 6.2741s ± 0.0559  | 6.2534s ± 0.0547  | 4.0518s ± 0.0259  |
| resnet152          | 8.8164s ± 0.0200  | 8.7882s ± 0.0208  | 5.6739s ± 0.0061  |
| squeezenet1_0      | 1.0822s ± 0.0055  | 1.0471s ± 0.0037  | 1.1011s ± 0.0032  |
| squeezenet1_1      | 0.5908s ± 0.0054  | 0.5672s ± 0.0032  | 0.5983s ± 0.0025  |
| densenet121        | 4.5867s ± 0.0327  | 4.5633s ± 0.0221  | 4.6789s ± 0.0094  |
| densenet161        | 9.1921s ± 0.0302  | 9.1772s ± 0.0218  | 8.9775s ± 0.0437  |
| densenet169        | 5.2167s ± 0.0757  | 5.1627s ± 0.0132  | 5.5359s ± 0.0128  |
| densenet201        | 6.7485s ± 0.0155  | 6.7488s ± 0.0234  | 7.3500s ± 0.0180  |
| inception_v3       | 2.0391s ± 0.0113  | 2.0031s ± 0.0022  | 1.2050s ± 0.0109  |
| googlenet          | 1.4318s ± 0.0599  | 1.3352s ± 0.0056  | 0.5418s ± 0.0067  |
| shufflenet_v2_x0_5 | 0.3859s ± 0.0011  | 0.3811s ± 0.0012  | 0.3740s ± 0.0021  |
| shufflenet_v2_x1_0 | 0.4966s ± 0.0015  | 0.4921s ± 0.0026  | 0.4813s ± 0.0030  |
| shufflenet_v2_x1_5 | 0.7109s ± 0.0092  | 0.7034s ± 0.0023  | 0.6741s ± 0.0036  |
| shufflenet_v2_x2_0 | 1.0531s ± 0.0035  | 1.0435s ± 0.0027  | 0.9790s ± 0.0030  |
| mobilenet_v2       | 2.6147s ± 0.0247  | 2.6081s ± 0.0112  | 2.2661s ± 0.0361  |
| mobilenet_v3_small | 0.6631s ± 0.0038  | 0.6627s ± 0.0028  | 0.6472s ± 0.0102  |
| mobilenet_v3_large | 1.8188s ± 0.0347  | 1.7902s ± 0.0245  | 1.6180s ± 0.0234  |
| resnext50_32x4d    | 4.8912s ± 0.0083  | 4.8711s ± 0.0072  | 3.8442s ± 0.0227  |
| resnext101_32x8d   | 12.2317s ± 0.0295 | 12.2523s ± 0.0521 | 9.2784s ± 0.0107  |
| wide_resnet50_2    | 6.3667s ± 0.0393  | 6.4739s ± 0.0139  | 3.2907s ± 0.0076  |
| wide_resnet101_2   | 10.3771s ± 0.0238 | 10.8201s ± 0.0155 | 4.7608s ± 0.0198  |
| mnasnet0_5         | 1.1845s ± 0.0231  | 1.1803s ± 0.0193  | 1.1389s ± 0.0270  |
| mnasnet0_75        | 2.0179s ± 0.0466  | 2.0364s ± 0.0260  | 1.8216s ± 0.0559  |
| mnasnet1_0         | 2.3628s ± 0.0295  | 2.3177s ± 0.0406  | 2.1792s ± 0.0113  |
| mnasnet1_3         | 3.2251s ± 0.0606  | 3.1708s ± 0.0701  | 2.8458s ± 0.0548  |
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