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
Update timestamp 08/07/2021 15:13:36

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2611s ± 0.0075  | 0.2518s ± 0.0037  | 0.1100s ± 0.0022  |
| vgg11              | 2.7949s ± 0.0033  | 2.7520s ± 0.0040  | 1.2218s ± 0.0063  |
| vgg11_bn           | 3.7092s ± 0.0076  | 3.6847s ± 0.0130  | 1.2133s ± 0.0038  |
| vgg13              | 4.2121s ± 0.0084  | 4.1811s ± 0.0283  | 1.8762s ± 0.0066  |
| vgg13_bn           | 5.7312s ± 0.0063  | 5.6968s ± 0.0155  | 1.8799s ± 0.0042  |
| vgg16              | 5.2460s ± 0.0202  | 5.2196s ± 0.0067  | 2.2113s ± 0.0075  |
| vgg16_bn           | 6.9022s ± 0.0056  | 6.8923s ± 0.0343  | 2.2177s ± 0.0200  |
| vgg19              | 6.3172s ± 0.0636  | 6.2776s ± 0.0062  | 2.5511s ± 0.0071  |
| vgg19_bn           | 8.0935s ± 0.0489  | 8.0356s ± 0.0123  | 2.5188s ± 0.0058  |
| resnet18           | 1.1003s ± 0.0123  | 1.0879s ± 0.0161  | 0.6511s ± 0.0037  |
| resnet34           | 1.7084s ± 0.0118  | 1.6862s ± 0.0105  | 0.9661s ± 0.0034  |
| resnet50           | 4.1097s ± 0.0114  | 4.0906s ± 0.0045  | 2.6223s ± 0.0241  |
| resnet101          | 6.0550s ± 0.0141  | 6.0416s ± 0.0161  | 3.8089s ± 0.0071  |
| resnet152          | 8.4895s ± 0.0413  | 8.4617s ± 0.0139  | 5.2914s ± 0.0057  |
| squeezenet1_0      | 1.0759s ± 0.0050  | 1.0392s ± 0.0031  | 1.1342s ± 0.0050  |
| squeezenet1_1      | 0.5903s ± 0.0030  | 0.5685s ± 0.0037  | 0.6218s ± 0.0022  |
| densenet121        | 4.6373s ± 0.0126  | 4.6209s ± 0.0137  | 4.7464s ± 0.0180  |
| densenet161        | 8.8009s ± 0.1452  | 8.7450s ± 0.0276  | 8.5676s ± 0.0334  |
| densenet169        | 5.0044s ± 0.0909  | 4.9495s ± 0.0177  | 5.2754s ± 0.0064  |
| densenet201        | 6.4589s ± 0.0274  | 6.4183s ± 0.0133  | 6.9760s ± 0.0307  |
| inception_v3       | 1.9811s ± 0.0072  | 1.9481s ± 0.0045  | 1.1542s ± 0.0183  |
| googlenet          | 1.4574s ± 0.0364  | 1.3628s ± 0.0051  | 0.5542s ± 0.0057  |
| shufflenet_v2_x0_5 | 0.3868s ± 0.0010  | 0.3984s ± 0.0281  | 0.3867s ± 0.0049  |
| shufflenet_v2_x1_0 | 0.4868s ± 0.0032  | 0.4824s ± 0.0040  | 0.4867s ± 0.0026  |
| shufflenet_v2_x1_5 | 0.6907s ± 0.0066  | 0.6843s ± 0.0032  | 0.5848s ± 0.0033  |
| shufflenet_v2_x2_0 | 1.0242s ± 0.0043  | 1.0174s ± 0.0047  | 0.8441s ± 0.0034  |
| mobilenet_v2       | 2.5118s ± 0.0638  | 2.5418s ± 0.0538  | 2.2880s ± 0.0373  |
| mobilenet_v3_small | 0.6960s ± 0.0056  | 0.6946s ± 0.0047  | 0.6680s ± 0.0100  |
| mobilenet_v3_large | 1.7250s ± 0.0295  | 1.7663s ± 0.0275  | 1.6494s ± 0.0133  |
| resnext50_32x4d    | 4.6926s ± 0.0211  | 4.6933s ± 0.0178  | 3.8132s ± 0.0059  |
| resnext101_32x8d   | 11.7946s ± 0.0921 | 11.7650s ± 0.0561 | 8.7537s ± 0.0462  |
| wide_resnet50_2    | 6.3129s ± 0.0156  | 6.3038s ± 0.0175  | 3.2122s ± 0.0094  |
| wide_resnet101_2   | 10.1306s ± 0.0832 | 10.1043s ± 0.0168 | 4.5457s ± 0.0143  |
| mnasnet0_5         | 1.2819s ± 0.0060  | 1.2816s ± 0.0173  | 1.1813s ± 0.0054  |
| mnasnet0_75        | 2.0209s ± 0.0395  | 2.0153s ± 0.0395  | 1.7384s ± 0.0077  |
| mnasnet1_0         | 2.3863s ± 0.0463  | 2.3546s ± 0.0448  | 2.1650s ± 0.0228  |
| mnasnet1_3         | 3.2632s ± 0.0567  | 3.2442s ± 0.0387  | 2.7471s ± 0.0565  |
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