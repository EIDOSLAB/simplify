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
Update timestamp 08/10/2021 12:09:53

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0062s ± 0.0004 | 0.0069s ± 0.0006 | 0.0030s ± 0.0000  |
| densenet121        | 0.0318s ± 0.0019 | 0.0336s ± 0.0039 | 0.0253s ± 0.0115  |
| googlenet          | 0.0159s ± 0.0029 | 0.0155s ± 0.0028 | 0.0106s ± 0.0023  |
| inception_v3       | 0.0294s ± 0.0138 | 0.0219s ± 0.0032 | 0.0116s ± 0.0001  |
| mnasnet1_0         | 0.0150s ± 0.0010 | 0.0130s ± 0.0007 | 0.0191s ± 0.0116  |
| mobilenet_v3_large | 0.0155s ± 0.0017 | 0.0122s ± 0.0010 | 0.0122s ± 0.0007  |
| resnet50           | 0.0255s ± 0.0115 | 0.0261s ± 0.0096 | 0.0167s ± 0.0015  |
| resnext101_32x8d   | 0.0809s ± 0.0211 | 0.0893s ± 0.0268 | 0.0729s ± 0.0155  |
| shufflenet_v2_x2_0 | 0.0201s ± 0.0013 | 0.0196s ± 0.0082 | 0.0144s ± 0.0008  |
| squeezenet1_1      | 0.0050s ± 0.0009 | 0.0050s ± 0.0006 | 0.0038s ± 0.0002  |
| vgg19_bn           | 0.0381s ± 0.0079 | 0.0357s ± 0.0004 | 0.0137s ± 0.0038  |
| wide_resnet101_2   | 0.0795s ± 0.0226 | 0.0732s ± 0.0139 | 0.0599s ± 0.0115  |
<!-- benchmark eval ends -->

Training mode (keeps BatchNorm)

<!-- benchmark train starts -->
Update timestamp 08/10/2021 12:12:18

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0060s ± 0.0001 | 0.0060s ± 0.0001 | 0.0031s ± 0.0000  |
| densenet121        | 0.0533s ± 0.0044 | 0.0530s ± 0.0046 | 0.0421s ± 0.0085  |
| googlenet          | 0.0202s ± 0.0022 | 0.0211s ± 0.0046 | 0.0145s ± 0.0017  |
| inception_v3       | 0.0355s ± 0.0058 | 0.0381s ± 0.0077 | 0.0233s ± 0.0015  |
| mnasnet1_0         | 0.0206s ± 0.0012 | 0.0235s ± 0.0026 | 0.0215s ± 0.0006  |
| mobilenet_v3_large | 0.0182s ± 0.0025 | 0.0193s ± 0.0017 | 0.0208s ± 0.0018  |
| resnet50           | 0.0356s ± 0.0084 | 0.0354s ± 0.0043 | 0.0310s ± 0.0031  |
| resnext101_32x8d   | 0.1073s ± 0.0135 | 0.1088s ± 0.0192 | 0.0989s ± 0.0093  |
| shufflenet_v2_x2_0 | 0.0241s ± 0.0022 | 0.0241s ± 0.0015 | 0.0240s ± 0.0018  |
| squeezenet1_1      | 0.0047s ± 0.0001 | 0.0047s ± 0.0000 | 0.0038s ± 0.0001  |
| vgg19_bn           | 0.0410s ± 0.0161 | 0.0404s ± 0.0115 | 0.0178s ± 0.0035  |
| wide_resnet101_2   | 0.0968s ± 0.0142 | 0.0969s ± 0.0157 | 0.0733s ± 0.0090  |
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