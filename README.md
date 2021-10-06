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

<!-- benchmark starts -->
Update timestamp 06/10/2021 09:27:38

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0207s ± 0.0013 | 0.0204s ± 0.0004 | 0.0200s ± 0.0009  |
| densenet121        | 0.1885s ± 0.0016 | 0.1898s ± 0.0023 | 0.1261s ± 0.0032  |
| googlenet          | 0.1009s ± 0.0010 | 0.0897s ± 0.0032 | 0.0436s ± 0.0039  |
| inception_v3       | 0.1429s ± 0.0015 | 0.1405s ± 0.0078 | 0.0898s ± 0.0041  |
| mnasnet1_0         | 0.2231s ± 0.0013 | 0.2262s ± 0.0025 | 0.2245s ± 0.0167  |
| mobilenet_v3_large | 0.1356s ± 0.0014 | 0.1404s ± 0.0018 | 0.1256s ± 0.0039  |
| resnet50           | 0.2124s ± 0.0035 | 0.2098s ± 0.0023 | 0.1077s ± 0.0043  |
| resnext101_32x8d   | 1.3789s ± 0.0329 | 1.3571s ± 0.0453 | 0.7188s ± 0.0290  |
| shufflenet_v2_x2_0 | 0.0777s ± 0.0020 | 0.0765s ± 0.0028 | 0.0512s ± 0.0012  |
| squeezenet1_1      | 0.0364s ± 0.0015 | 0.0335s ± 0.0006 | 0.0285s ± 0.0005  |
| vgg19_bn           | 0.1141s ± 0.0046 | 0.1110s ± 0.0066 | 0.0487s ± 0.0023  |
| wide_resnet101_2   | 0.7690s ± 0.0133 | 0.7474s ± 0.0149 | 0.2902s ± 0.0109  |
<!-- benchmark ends -->

Training mode (leaves BatchNorm)

<!-- benchmark starts -->
Update timestamp 06/10/2021 09:17:52

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0229s ± 0.0140 | 0.0189s ± 0.0066 | 0.0175s ± 0.0017  |
| densenet121        | 0.1849s ± 0.0115 | 0.2315s ± 0.0104 | 0.2098s ± 0.0087  |
| googlenet          | 0.1008s ± 0.0011 | 0.1014s ± 0.0011 | 0.0465s ± 0.0005  |
| inception_v3       | 0.1408s ± 0.0047 | 0.1606s ± 0.0028 | 0.1197s ± 0.0053  |
| mnasnet1_0         | 0.2214s ± 0.0018 | 0.2419s ± 0.0013 | 0.2214s ± 0.0125  |
| mobilenet_v3_large | 0.1352s ± 0.0017 | 0.1450s ± 0.0012 | 0.1418s ± 0.0063  |
| resnet50           | 0.2187s ± 0.0025 | 0.2217s ± 0.0079 | 0.1223s ± 0.0071  |
| resnext101_32x8d   | 1.4764s ± 0.1845 | 1.4690s ± 0.0527 | 1.3259s ± 0.0517  |
| shufflenet_v2_x2_0 | 0.0786s ± 0.0025 | 0.1038s ± 0.0028 | 0.0744s ± 0.0025  |
| squeezenet1_1      | 0.0362s ± 0.0013 | 0.0350s ± 0.0003 | 0.0356s ± 0.0012  |
| vgg19_bn           | 0.1137s ± 0.0022 | 0.1004s ± 0.0017 | 0.0469s ± 0.0021  |
| wide_resnet101_2   | 0.7587s ± 0.0107 | 0.8054s ± 0.0613 | 0.3695s ± 0.0168  |
<!-- benchmark ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


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