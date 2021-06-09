# simplify

[![tests](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml/badge.svg)](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml)

<!-- benchmark starts -->
Update timestamp 09/06/2021 15:38:13

| Architecture       | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|
| alexnet            | 0.2731s ± 0.0092  | 0.1254s ± 0.0016  |
| vgg11              | 2.9845s ± 0.1228  | 1.3543s ± 0.1091  |
| vgg11_bn           | 4.0430s ± 0.1696  | 1.3019s ± 0.0517  |
| vgg13              | 4.5410s ± 0.2176  | 1.9430s ± 0.0652  |
| vgg13_bn           | 6.0788s ± 0.2389  | 2.0046s ± 0.0792  |
| vgg16              | 5.7167s ± 0.1638  | 2.3854s ± 0.1181  |
| vgg16_bn           | 7.4610s ± 0.1426  | 2.4199s ± 0.0945  |
| vgg19              | 6.9993s ± 0.2536  | 2.7182s ± 0.0908  |
| vgg19_bn           | 8.6919s ± 0.2255  | 2.7051s ± 0.0884  |
| resnet18           | 1.2333s ± 0.0803  | 0.7089s ± 0.0613  |
| resnet34           | 2.0171s ± 0.1131  | 1.1020s ± 0.0755  |
| resnet50           | 4.2291s ± 0.1779  | 2.5752s ± 0.0344  |
| resnet101          | 6.6829s ± 0.2063  | 3.8900s ± 0.0553  |
| resnet152          | 9.5327s ± 0.3259  | 5.3997s ± 0.0327  |
| squeezenet1_0      | 1.1222s ± 0.0603  | 0.6429s ± 0.0376  |
| squeezenet1_1      | 0.5950s ± 0.0126  | 0.3481s ± 0.0072  |
| inception_v3       | 2.1750s ± 0.0880  | 0.7519s ± 0.0102  |
| googlenet          | 1.6995s ± 0.1028  | 0.6287s ± 0.0089  |
| shufflenet_v2_x0_5 | 0.4654s ± 0.0321  | 0.4422s ± 0.0201  |
| shufflenet_v2_x1_0 | 0.5513s ± 0.0151  | 0.5403s ± 0.0139  |
| shufflenet_v2_x1_5 | 0.7795s ± 0.0097  | 0.7340s ± 0.0104  |
| shufflenet_v2_x2_0 | 1.2731s ± 0.0286  | 1.0746s ± 0.0271  |
| mobilenet_v2       | 2.8693s ± 0.0477  | 2.5030s ± 0.0976  |
| mobilenet_v3_small | 0.8012s ± 0.0258  | 0.7673s ± 0.0208  |
| mobilenet_v3_large | 1.9958s ± 0.0621  | 1.8270s ± 0.0386  |
| resnext50_32x4d    | 5.2487s ± 0.1776  | 3.7297s ± 0.0611  |
| resnext101_32x8d   | 13.0387s ± 0.2461 | 8.8348s ± 0.0634  |
| wide_resnet50_2    | 6.9899s ± 0.2028  | 3.2065s ± 0.0522  |
| wide_resnet101_2   | 12.0117s ± 0.5655 | 4.6933s ± 0.0971  |
| mnasnet0_5         | 1.4294s ± 0.0369  | 1.3574s ± 0.0322  |
| mnasnet0_75        | 2.3028s ± 0.0594  | 2.1209s ± 0.0551  |
| mnasnet1_0         | 2.7339s ± 0.0399  | 2.4632s ± 0.0396  |
| mnasnet1_3         | 3.7064s ± 0.0867  | 3.2814s ± 0.0525  |
<!-- benchmark ends -->

### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one

<!-- table starts -->
Update timestamp 09/06/2021 14:26:11

|    Architecture    |  BatchNorm Folding  |  Bias Propagation  |   Simplification   |  Grouping  |
|--------------------|---------------------|--------------------|--------------------|------------|
|      alexnet       | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|       vgg11        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      vgg11_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|       vgg13        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      vgg13_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|       vgg16        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      vgg16_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|       vgg19        | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      vgg19_bn      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      resnet18      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      resnet34      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     resnet101      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     resnet152      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|   squeezenet1_0    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |   False    |
|    densenet161     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    densenet169     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |   False    |
|    densenet201     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |   False    |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| shufflenet_v2_x0_5 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| shufflenet_v2_x1_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| shufflenet_v2_x1_5 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    mobilenet_v2    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| mobilenet_v3_small | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  resnext50_32x4d   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  wide_resnet50_2   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     mnasnet0_5     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    mnasnet0_75     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     mnasnet1_3     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
<!-- table ends -->