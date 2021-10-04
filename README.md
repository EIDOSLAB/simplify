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
Update timestamp 09/07/2021 11:05:07

Random structured pruning amount = 50.0%

| Architecture       | Dense time       | Pruned time      | Simplified time   |
|--------------------|------------------|------------------|-------------------|
| alexnet            | 0.0138s ± 0.0050 | 0.0120s ± 0.0004 | 0.0080s ± 0.0002  |
| vgg11              | 0.0552s ± 0.0010 | 0.0525s ± 0.0014 | 0.0263s ± 0.0004  |
| vgg11_bn           | 0.0597s ± 0.0003 | 0.0547s ± 0.0015 | 0.0258s ± 0.0007  |
| vgg13              | 0.0658s ± 0.0004 | 0.0626s ± 0.0015 | 0.0295s ± 0.0011  |
| vgg13_bn           | 0.0731s ± 0.0038 | 0.0687s ± 0.0029 | 0.0299s ± 0.0007  |
| vgg16              | 0.0767s ± 0.0015 | 0.0724s ± 0.0014 | 0.0316s ± 0.0007  |
| vgg16_bn           | 0.0831s ± 0.0025 | 0.0784s ± 0.0014 | 0.0323s ± 0.0008  |
| vgg19              | 0.0869s ± 0.0025 | 0.0850s ± 0.0012 | 0.0354s ± 0.0009  |
| vgg19_bn           | 0.0965s ± 0.0038 | 0.0873s ± 0.0016 | 0.0344s ± 0.0006  |
| resnet18           | 0.0153s ± 0.0005 | 0.0142s ± 0.0004 | 0.0111s ± 0.0003  |
| resnet34           | 0.0231s ± 0.0001 | 0.0223s ± 0.0006 | 0.0164s ± 0.0003  |
| resnet50           | 0.0552s ± 0.0012 | 0.0543s ± 0.0004 | 0.0341s ± 0.0004  |
| resnet101          | 0.0959s ± 0.0001 | 0.0958s ± 0.0006 | 0.0539s ± 0.0004  |
| resnet152          | 0.1388s ± 0.0031 | 0.1377s ± 0.0009 | 0.0755s ± 0.0011  |
| squeezenet1_0      | 0.0199s ± 0.0005 | 0.0171s ± 0.0004 | 0.0164s ± 0.0003  |
| squeezenet1_1      | 0.0140s ± 0.0002 | 0.0123s ± 0.0000 | 0.0114s ± 0.0002  |
| densenet121        | 0.0545s ± 0.0002 | 0.0524s ± 0.0005 | 0.0468s ± 0.0005  |
| densenet161        | 0.1275s ± 0.0014 | 0.1268s ± 0.0008 | 0.0987s ± 0.0007  |
| densenet169        | 0.0714s ± 0.0010 | 0.0699s ± 0.0009 | 0.0612s ± 0.0005  |
| densenet201        | 0.0916s ± 0.0001 | 0.0905s ± 0.0004 | 0.0769s ± 0.0005  |
| inception_v3       | 0.0450s ± 0.0010 | 0.0426s ± 0.0004 | 0.0304s ± 0.0003  |
| googlenet          | 0.0437s ± 0.0007 | 0.0371s ± 0.0002 | 0.0206s ± 0.0002  |
| shufflenet_v2_x0_5 | 0.0170s ± 0.0003 | 0.0166s ± 0.0002 | 0.0168s ± 0.0002  |
| shufflenet_v2_x1_0 | 0.0110s ± 0.0004 | 0.0105s ± 0.0001 | 0.0100s ± 0.0001  |
| shufflenet_v2_x1_5 | 0.0138s ± 0.0004 | 0.0132s ± 0.0001 | 0.0105s ± 0.0002  |
| shufflenet_v2_x2_0 | 0.0209s ± 0.0003 | 0.0205s ± 0.0005 | 0.0150s ± 0.0002  |
| mobilenet_v2       | 0.0872s ± 0.0002 | 0.0874s ± 0.0006 | 0.0891s ± 0.0004  |
| mobilenet_v3_small | 0.0323s ± 0.0003 | 0.0323s ± 0.0002 | 0.0327s ± 0.0003  |
| mobilenet_v3_large | 0.0606s ± 0.0003 | 0.0609s ± 0.0010 | 0.0612s ± 0.0004  |
| resnext50_32x4d    | 0.0819s ± 0.0018 | 0.0802s ± 0.0004 | 0.0562s ± 0.0006  |
| resnext101_32x8d   | 0.2804s ± 0.0057 | 0.2775s ± 0.0024 | 0.1709s ± 0.0009  |
| wide_resnet50_2    | 0.1150s ± 0.0011 | 0.1135s ± 0.0012 | 0.0522s ± 0.0007  |
| wide_resnet101_2   | 0.2148s ± 0.0020 | 0.2138s ± 0.0022 | 0.0822s ± 0.0011  |
| mnasnet0_5         | 0.0529s ± 0.0004 | 0.0528s ± 0.0003 | 0.0530s ± 0.0003  |
| mnasnet0_75        | 0.0810s ± 0.0004 | 0.0813s ± 0.0003 | 0.0796s ± 0.0004  |
| mnasnet1_0         | 0.1016s ± 0.0004 | 0.1016s ± 0.0004 | 0.1012s ± 0.0005  |
| mnasnet1_3         | 0.1344s ± 0.0006 | 0.1345s ± 0.0004 | 0.1313s ± 0.0005  |

<!-- benchmark ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 04/10/2021 12:39:20

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
<!-- table ends -->
</details>