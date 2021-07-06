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
Update timestamp 06/07/2021 12:36:12

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2525s ± 0.0090  | 0.2443s ± 0.0020  | 0.1105s ± 0.0009  |
| vgg11              | 2.7922s ± 0.0067  | 2.7607s ± 0.0084  | 1.2218s ± 0.0150  |
| vgg11_bn           | 3.7156s ± 0.0077  | 3.6826s ± 0.0074  | 1.2041s ± 0.0042  |
| vgg13              | 4.2280s ± 0.0115  | 4.1725s ± 0.0077  | 1.8669s ± 0.0074  |
| vgg13_bn           | 5.7212s ± 0.0089  | 5.6758s ± 0.0083  | 1.8711s ± 0.0066  |
| vgg16              | 5.2454s ± 0.0125  | 5.2256s ± 0.0215  | 2.2005s ± 0.0036  |
| vgg16_bn           | 6.9232s ± 0.0117  | 6.8650s ± 0.0103  | 2.2089s ± 0.0054  |
| vgg19              | 6.3111s ± 0.0405  | 6.3031s ± 0.0666  | 2.5333s ± 0.0047  |
| vgg19_bn           | 8.0813s ± 0.0152  | 8.0368s ± 0.0191  | 2.5475s ± 0.0127  |
| resnet18           | 1.0800s ± 0.0103  | 1.0719s ± 0.0057  | 0.6358s ± 0.0029  |
| resnet34           | 1.8251s ± 0.0052  | 1.8206s ± 0.0174  | 0.9768s ± 0.0047  |
| resnet50           | 4.0916s ± 0.0039  | 4.0842s ± 0.0190  | 2.6366s ± 0.0127  |
| resnet101          | 6.2351s ± 0.0055  | 6.2279s ± 0.0251  | 4.0378s ± 0.0136  |
| resnet152          | 8.7859s ± 0.0204  | 8.7689s ± 0.0236  | 5.6655s ± 0.0334  |
| squeezenet1_0      | 1.0808s ± 0.0044  | 1.0427s ± 0.0037  | 1.1004s ± 0.0073  |
| squeezenet1_1      | 0.6032s ± 0.0029  | 0.5847s ± 0.0074  | 0.6126s ± 0.0041  |
| densenet121        | 4.6245s ± 0.0301  | 4.6049s ± 0.0303  | 4.6357s ± 0.0059  |
| densenet161        | 8.8236s ± 0.1377  | 8.7528s ± 0.0159  | 8.4635s ± 0.0237  |
| densenet169        | 5.1807s ± 0.0496  | 5.1484s ± 0.0224  | 5.2525s ± 0.0201  |
| densenet201        | 6.7767s ± 0.0540  | 6.7773s ± 0.0189  | 7.0328s ± 0.0074  |
| inception_v3       | 1.9906s ± 0.0098  | 1.9537s ± 0.0017  | 1.1898s ± 0.0068  |
| googlenet          | 1.5503s ± 0.0058  | 1.4651s ± 0.0040  | 0.5864s ± 0.0063  |
| shufflenet_v2_x0_5 | 0.3851s ± 0.0012  | 0.3922s ± 0.0039  | 0.3862s ± 0.0068  |
| shufflenet_v2_x1_0 | 0.5067s ± 0.0050  | 0.4988s ± 0.0038  | 0.4821s ± 0.0044  |
| shufflenet_v2_x1_5 | 0.6980s ± 0.0042  | 0.6882s ± 0.0034  | 0.5656s ± 0.0039  |
| shufflenet_v2_x2_0 | 1.1740s ± 0.0280  | 1.1776s ± 0.0034  | 0.8466s ± 0.0030  |
| mobilenet_v2       | 2.6071s ± 0.0149  | 2.5654s ± 0.0487  | 2.2437s ± 0.0434  |
| mobilenet_v3_small | 0.6693s ± 0.0085  | 0.6562s ± 0.0076  | 0.6567s ± 0.0075  |
| mobilenet_v3_large | 1.8099s ± 0.0344  | 1.7999s ± 0.0381  | 1.5885s ± 0.0305  |
| resnext50_32x4d    | 4.9153s ± 0.0655  | 4.8743s ± 0.0159  | 3.7275s ± 0.0077  |
| resnext101_32x8d   | 11.9513s ± 0.0368 | 11.9250s ± 0.0265 | 9.2229s ± 0.0237  |
| wide_resnet50_2    | 6.3043s ± 0.0660  | 6.3065s ± 0.0536  | 3.1919s ± 0.0131  |
| wide_resnet101_2   | 10.1800s ± 0.0228 | 10.1543s ± 0.0198 | 4.6918s ± 0.0359  |
| mnasnet0_5         | 1.2298s ± 0.0237  | 1.2297s ± 0.0244  | 1.1170s ± 0.0039  |
| mnasnet0_75        | 1.9250s ± 0.0452  | 1.9398s ± 0.0383  | 1.6096s ± 0.0292  |
| mnasnet1_0         | 2.2882s ± 0.0337  | 2.3542s ± 0.0248  | 2.1556s ± 0.0295  |
| mnasnet1_3         | 3.1382s ± 0.0796  | 3.1204s ± 0.0770  | 2.5568s ± 0.0649  |
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