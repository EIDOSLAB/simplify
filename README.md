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

Evaluation mode (fuses BatchNorm)

<!-- benchmark eval starts -->
Update timestamp 07/10/2021 10:49:31

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0140s ± 0.0039 | 0.0188s ± 0.0049 | 0.0127s ± 0.0074  |
| densenet121        | 0.0691s ± 0.0073 | 0.0739s ± 0.0062 | 0.0472s ± 0.0067  |
| googlenet          | 0.0287s ± 0.0009 | 0.0286s ± 0.0011 | 0.0189s ± 0.0013  |
| inception_v3       | 0.0486s ± 0.0141 | 0.0520s ± 0.0018 | 0.0257s ± 0.0008  |
| mnasnet1_0         | 0.0295s ± 0.0037 | 0.0285s ± 0.0034 | 0.0309s ± 0.0017  |
| mobilenet_v3_large | 0.0275s ± 0.0025 | 0.0272s ± 0.0023 | 0.0275s ± 0.0014  |
| resnet50           | 0.0646s ± 0.0100 | 0.0597s ± 0.0022 | 0.0384s ± 0.0020  |
| resnext101_32x8d   | 0.2048s ± 0.0082 | 0.2047s ± 0.0106 | 0.1743s ± 0.0132  |
| shufflenet_v2_x2_0 | 0.0481s ± 0.0051 | 0.0470s ± 0.0041 | 0.0346s ± 0.0044  |
| squeezenet1_1      | 0.0159s ± 0.0025 | 0.0167s ± 0.0017 | 0.0102s ± 0.0012  |
| vgg19_bn           | 0.0961s ± 0.0041 | 0.0948s ± 0.0015 | 0.0377s ± 0.0007  |
| wide_resnet101_2   | 0.1722s ± 0.0368 | 0.1410s ± 0.0117 | 0.1037s ± 0.0067  |
<!-- benchmark eval ends -->

Training mode (keeps BatchNorm)

<!-- benchmark train starts -->
Update timestamp 07/10/2021 10:54:36

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0154s ± 0.0004 | 0.0156s ± 0.0004 | 0.0121s ± 0.0005  |
| densenet121        | 0.1163s ± 0.0058 | 0.1029s ± 0.0063 | 0.0795s ± 0.0043  |
| googlenet          | 0.0368s ± 0.0014 | 0.0367s ± 0.0013 | 0.0269s ± 0.0019  |
| inception_v3       | 0.0759s ± 0.0023 | 0.0763s ± 0.0021 | 0.0410s ± 0.0007  |
| mnasnet1_0         | 0.0576s ± 0.0047 | 0.0577s ± 0.0048 | 0.0539s ± 0.0048  |
| mobilenet_v3_large | 0.0511s ± 0.0035 | 0.0491s ± 0.0024 | 0.0382s ± 0.0020  |
| resnet50           | 0.0753s ± 0.0015 | 0.0874s ± 0.0036 | 0.0594s ± 0.0026  |
| resnext101_32x8d   | 0.2540s ± 0.0095 | 0.2507s ± 0.0062 | 0.1993s ± 0.0042  |
| shufflenet_v2_x2_0 | 0.0559s ± 0.0044 | 0.0568s ± 0.0035 | 0.0611s ± 0.0058  |
| squeezenet1_1      | 0.0113s ± 0.0013 | 0.0103s ± 0.0004 | 0.0077s ± 0.0002  |
| vgg19_bn           | 0.0617s ± 0.0012 | 0.0658s ± 0.0018 | 0.0417s ± 0.0031  |
| wide_resnet101_2   | 0.1803s ± 0.0077 | 0.1856s ± 0.0142 | 0.1318s ± 0.0053  |
<!-- benchmark train ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one

Fuse BatchNorm

<!-- table fuse starts -->
Update timestamp 06/10/2021 20:26:15

|    Architecture    |  BatchNorm Folding  |  Bias Propagation  |   Simplification   |
|--------------------|---------------------|--------------------|--------------------|
|      alexnet       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg19_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |

<!-- table fuse ends -->

Keep BatchNorm

<!-- table no fuse starts -->
Update timestamp 06/10/2021 20:36:11

|    Architecture    |  BatchNorm Folding  |  Bias Propagation  |   Simplification   |
|--------------------|---------------------|--------------------|--------------------|
|      alexnet       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|      vgg19_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |

<!-- table no fuse ends -->
</details>