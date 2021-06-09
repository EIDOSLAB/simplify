# simplify

[![tests](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml/badge.svg)](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml)

<!-- benchmark starts -->
Update timestamp 08/06/2021 16:11:45

| Architecture       | Pruned time       | Simplified time   |
|--------------------|-------------------|-------------------|
| alexnet            | 0.2655s ± 0.0096  | 0.1218s ± 0.0016  |
| vgg11              | 3.1338s ± 0.2783  | 1.3406s ± 0.1123  |
| vgg11_bn           | 3.9512s ± 0.1491  | 0.5793s ± 0.0369  |
| vgg13              | 4.6584s ± 0.2138  | 1.9851s ± 0.1072  |
| vgg13_bn           | 6.0304s ± 0.2281  | 0.9474s ± 0.0361  |
| vgg16              | 5.7435s ± 0.1952  | 2.3785s ± 0.0809  |
| vgg16_bn           | 7.1335s ± 0.1791  | 1.0399s ± 0.0527  |
| vgg19              | 6.8106s ± 0.2464  | 2.8280s ± 0.1250  |
| vgg19_bn           | 8.7968s ± 0.1859  | 1.2263s ± 0.0489  |
| resnet18           | 1.2264s ± 0.1198  | 0.5284s ± 0.0063  |
| resnet34           | 2.0966s ± 0.1774  | 0.7313s ± 0.0103  |
| resnet50           | 4.4121s ± 0.1099  | 2.0185s ± 0.0228  |
| resnet101          | 6.6095s ± 0.2972  | 2.8979s ± 0.0403  |
| resnet152          | 8.9557s ± 0.2454  | 4.0231s ± 0.0315  |
| squeezenet1_0      | 1.1114s ± 0.0740  | 0.5961s ± 0.0071  |
| squeezenet1_1      | 0.5994s ± 0.0211  | 0.3473s ± 0.0014  |
| densenet121        | 4.6049s ± 0.1344  | 1.5496s ± 0.0245  |
| densenet161        | 9.4213s ± 0.1464  | 3.0602s ± 0.0468  |
| densenet169        | 5.5767s ± 0.0924  | 1.7422s ± 0.0709  |
| densenet201        | 6.8275s ± 0.0939  | 2.3306s ± 0.0290  |
| inception_v3       | 2.0719s ± 0.0569  | 0.3900s ± 0.0118  |
| googlenet          | 1.4923s ± 0.0995  | 0.3173s ± 0.0077  |
| shufflenet_v2_x0_5 | 0.4509s ± 0.0288  | 0.4154s ± 0.0259  |
| shufflenet_v2_x1_0 | 0.5311s ± 0.0064  | 0.4736s ± 0.0144  |
| shufflenet_v2_x1_5 | 0.7526s ± 0.0190  | 0.6492s ± 0.0065  |
| shufflenet_v2_x2_0 | 1.1105s ± 0.0170  | 0.9101s ± 0.0130  |
| mobilenet_v2       | 2.8251s ± 0.0280  | 2.3825s ± 0.0438  |
| mobilenet_v3_small | 0.7704s ± 0.0198  | 0.7375s ± 0.0122  |
| mobilenet_v3_large | 1.9998s ± 0.0351  | 1.7345s ± 0.0417  |
| resnext50_32x4d    | 5.3506s ± 0.2075  | 3.1867s ± 0.0417  |
| resnext101_32x8d   | 13.1396s ± 0.1637 | 7.3858s ± 0.0594  |
| wide_resnet50_2    | 7.1732s ± 0.2553  | 2.2192s ± 0.0292  |
| wide_resnet101_2   | 11.4132s ± 0.2258 | 3.2196s ± 0.0450  |
| mnasnet0_5         | 1.4374s ± 0.0451  | 1.3006s ± 0.0263  |
| mnasnet0_75        | 2.2768s ± 0.0472  | 1.9868s ± 0.0481  |
| mnasnet1_0         | 2.6419s ± 0.0527  | 2.3018s ± 0.0273  |
| mnasnet1_3         | 3.6410s ± 0.0518  | 3.0887s ± 0.0546  |

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