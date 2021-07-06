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
Update timestamp 06/07/2021 10:52:48

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2561s ± 0.0079  | 0.2507s ± 0.0038  | 0.1110s ± 0.0033  |
| vgg11              | 2.8165s ± 0.0542  | 2.7600s ± 0.0093  | 1.2264s ± 0.0328  |
| vgg11_bn           | 3.6924s ± 0.0077  | 3.6761s ± 0.0099  | 1.2094s ± 0.0037  |
| vgg13              | 4.2236s ± 0.0054  | 4.1783s ± 0.0086  | 1.8679s ± 0.0026  |
| vgg13_bn           | 5.7285s ± 0.0223  | 5.6877s ± 0.0131  | 1.8730s ± 0.0032  |
| vgg16              | 5.2544s ± 0.0268  | 5.2377s ± 0.0247  | 2.2139s ± 0.0132  |
| vgg16_bn           | 6.9378s ± 0.0318  | 6.8989s ± 0.0270  | 2.2067s ± 0.0035  |
| vgg19              | 6.3246s ± 0.0820  | 6.2894s ± 0.0148  | 2.5472s ± 0.0106  |
| vgg19_bn           | 8.1154s ± 0.0400  | 8.0594s ± 0.0253  | 2.5522s ± 0.0380  |
| resnet18           | 1.0650s ± 0.0106  | 1.0475s ± 0.0051  | 0.7795s ± 0.0058  |
| resnet34           | 1.7949s ± 0.0197  | 1.7803s ± 0.0087  | 1.1761s ± 0.0043  |
| resnet50           | 4.1049s ± 0.0364  | 4.1000s ± 0.0258  | 3.2525s ± 0.0233  |
| resnet101          | 6.0559s ± 0.0268  | 6.0581s ± 0.0434  | 4.7324s ± 0.0193  |
| resnet152          | 8.5037s ± 0.0903  | 8.8386s ± 0.0505  | 6.5174s ± 0.0049  |
| squeezenet1_0      | 1.0731s ± 0.0037  | 1.0420s ± 0.0037  | 1.3913s ± 0.0082  |
| squeezenet1_1      | 0.6023s ± 0.0037  | 0.5803s ± 0.0031  | 0.7312s ± 0.0025  |
| densenet121        | 4.6112s ± 0.0350  | 4.5794s ± 0.0106  | 5.0937s ± 0.0137  |
| densenet161        | 8.8832s ± 0.1316  | 8.8209s ± 0.0330  | 9.2521s ± 0.0420  |
| densenet169        | 5.1876s ± 0.0243  | 5.1680s ± 0.0174  | 5.5750s ± 0.0644  |
| densenet201        | 6.4218s ± 0.0157  | 6.3896s ± 0.0197  | 7.3679s ± 0.0271  |
| inception_v3       | 1.9713s ± 0.0092  | 1.9650s ± 0.0611  | 1.2391s ± 0.0095  |
| googlenet          | 1.4627s ± 0.0282  | 1.3722s ± 0.0043  | 0.5612s ± 0.0095  |
| shufflenet_v2_x0_5 | 0.3862s ± 0.0011  | 0.3816s ± 0.0006  | 0.4106s ± 0.0027  |
| shufflenet_v2_x1_0 | 0.4909s ± 0.0029  | 0.4844s ± 0.0029  | 0.5169s ± 0.0027  |
| shufflenet_v2_x1_5 | 0.6766s ± 0.0024  | 0.6682s ± 0.0029  | 0.6525s ± 0.0033  |
| shufflenet_v2_x2_0 | 1.1643s ± 0.0342  | 1.1669s ± 0.0033  | 1.0245s ± 0.0037  |
| mobilenet_v2       | 2.6460s ± 0.0626  | 2.6650s ± 0.0445  | 2.4821s ± 0.0457  |
| mobilenet_v3_small | 0.6785s ± 0.0089  | 0.6695s ± 0.0024  | 0.6970s ± 0.0056  |
| mobilenet_v3_large | 1.7617s ± 0.0064  | 1.7766s ± 0.0274  | 1.7634s ± 0.0040  |
| resnext50_32x4d    | 4.7387s ± 0.0129  | 4.7346s ± 0.0167  | 4.6261s ± 0.0067  |
| resnext101_32x8d   | 11.8021s ± 0.0350 | 11.8019s ± 0.0268 | 10.4213s ± 0.0165 |
| wide_resnet50_2    | 6.3425s ± 0.0380  | 6.3177s ± 0.0097  | 3.7876s ± 0.0112  |
| wide_resnet101_2   | 10.1397s ± 0.0345 | 10.5333s ± 0.0142 | 5.3807s ± 0.0250  |
| mnasnet0_5         | 1.2913s ± 0.0256  | 1.2652s ± 0.0222  | 1.2846s ± 0.0230  |
| mnasnet0_75        | 2.0083s ± 0.0383  | 1.9819s ± 0.0089  | 1.8171s ± 0.0179  |
| mnasnet1_0         | 2.3313s ± 0.0332  | 2.3186s ± 0.0060  | 2.3087s ± 0.0188  |
| mnasnet1_3         | 3.2669s ± 0.0468  | 3.2379s ± 0.0299  | 2.9650s ± 0.0550  |
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