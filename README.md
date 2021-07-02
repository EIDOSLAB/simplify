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
Update timestamp 01/07/2021 15:19:34

Random structured pruning amount = 50.0%

| Architecture       | Dense time        | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|-------------------|
| alexnet            | 0.2545s ± 0.0102  | 0.2460s ± 0.0033  | 0.1103s ± 0.0039  |
| vgg11              | 2.8058s ± 0.0051  | 2.8157s ± 0.1149  | 1.2167s ± 0.0062  |
| vgg11_bn           | 3.7297s ± 0.0190  | 3.6972s ± 0.0104  | 1.2141s ± 0.0121  |
| vgg13              | 4.2333s ± 0.0073  | 4.2162s ± 0.0544  | 1.8771s ± 0.0034  |
| vgg13_bn           | 5.7763s ± 0.0160  | 5.7112s ± 0.0207  | 1.8721s ± 0.0070  |
| vgg16              | 5.2540s ± 0.0148  | 5.2638s ± 0.1229  | 2.2055s ± 0.0111  |
| vgg16_bn           | 6.9439s ± 0.0177  | 6.8820s ± 0.0085  | 2.2065s ± 0.0060  |
| vgg19              | 6.3616s ± 0.1144  | 6.3080s ± 0.0054  | 2.5464s ± 0.0168  |
| vgg19_bn           | 8.1254s ± 0.0156  | 8.0577s ± 0.0218  | 2.5385s ± 0.0046  |
| resnet18           | 1.0742s ± 0.0161  | 1.0641s ± 0.0088  | 0.6485s ± 0.0046  |
| resnet34           | 1.7053s ± 0.0176  | 1.7012s ± 0.0220  | 0.9584s ± 0.0035  |
| resnet50           | 4.1101s ± 0.0121  | 4.1019s ± 0.0331  | 2.5838s ± 0.0037  |
| resnet101          | 6.2384s ± 0.0186  | 6.2269s ± 0.0151  | 3.8213s ± 0.0062  |
| resnet152          | 8.7623s ± 0.0339  | 8.7725s ± 0.0207  | 5.3389s ± 0.0046  |
| squeezenet1_0      | 1.0829s ± 0.0052  | 1.0507s ± 0.0049  | 1.1250s ± 0.0120  |
| squeezenet1_1      | 0.6039s ± 0.0041  | 0.5814s ± 0.0019  | 0.6173s ± 0.0065  |
| densenet121        | 4.6214s ± 0.0130  | 4.6042s ± 0.0098  | 4.7011s ± 0.0284  |
| densenet161        | 9.1962s ± 0.0789  | 9.1201s ± 0.0423  | 8.5322s ± 0.1125  |
| densenet169        | 4.9701s ± 0.0545  | 4.9325s ± 0.0091  | 5.1915s ± 0.0287  |
| densenet201        | 6.4126s ± 0.0179  | 6.3933s ± 0.0143  | 6.9536s ± 0.0311  |
| inception_v3       | 1.9772s ± 0.0087  | 1.9460s ± 0.0067  | 1.1785s ± 0.0068  |
| googlenet          | 1.4573s ± 0.0305  | 1.3629s ± 0.0044  | 0.5483s ± 0.0101  |
| shufflenet_v2_x0_5 | 0.3908s ± 0.0026  | 0.3854s ± 0.0029  | 0.3755s ± 0.0054  |
| shufflenet_v2_x1_0 | 0.4965s ± 0.0059  | 0.4897s ± 0.0026  | 0.4770s ± 0.0033  |
| shufflenet_v2_x1_5 | 0.7343s ± 0.0092  | 0.7265s ± 0.0021  | 0.6970s ± 0.0039  |
| shufflenet_v2_x2_0 | 1.0390s ± 0.0075  | 1.0301s ± 0.0036  | 0.9711s ± 0.0048  |
| mobilenet_v2       | 2.5646s ± 0.0211  | 2.6259s ± 0.0421  | 2.1950s ± 0.0118  |
| mobilenet_v3_small | 0.6707s ± 0.0076  | 0.6804s ± 0.0012  | 0.6513s ± 0.0116  |
| mobilenet_v3_large | 1.8021s ± 0.0211  | 1.7928s ± 0.0090  | 1.6201s ± 0.0203  |
| resnext50_32x4d    | 4.8371s ± 0.0162  | 4.8366s ± 0.0455  | 3.6868s ± 0.0071  |
| resnext101_32x8d   | 11.8295s ± 0.0205 | 11.8021s ± 0.0425 | 8.7308s ± 0.0104  |
| wide_resnet50_2    | 6.3901s ± 0.0896  | 6.3137s ± 0.0111  | 3.1445s ± 0.0219  |
| wide_resnet101_2   | 10.6236s ± 0.0380 | 10.5310s ± 0.0359 | 4.4830s ± 0.0211  |
| mnasnet0_5         | 1.2606s ± 0.0296  | 1.2623s ± 0.0166  | 1.1916s ± 0.0087  |
| mnasnet0_75        | 2.0221s ± 0.0347  | 2.0291s ± 0.0261  | 1.8622s ± 0.0131  |
| mnasnet1_0         | 2.4041s ± 0.0379  | 2.4090s ± 0.0365  | 2.1125s ± 0.0305  |
| mnasnet1_3         | 3.4027s ± 0.0776  | 3.3037s ± 0.0676  | 3.0261s ± 0.0116  |
<!-- benchmark ends -->

#### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one


<!-- table starts -->
Update timestamp 02/07/2021 12:21:41

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
| shufflenet_v2_x0_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| shufflenet_v2_x1_0 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| shufflenet_v2_x1_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|    mobilenet_v2    | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| mobilenet_v3_small | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|  resnext50_32x4d   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet50_2   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
|     mnasnet0_5     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|    mnasnet0_75     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
|     mnasnet1_3     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |
<!-- table ends -->
</details>