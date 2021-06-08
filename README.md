# simplify

[![tests](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml/badge.svg)](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml)

<!-- benchmark starts -->
Update timestamp 03/06/2021 18:18:16

| Architecture       | Pruned time     | Simplified time (p=50%)   |
|--------------------|-----------------|---------------------------|
| alexnet            | 0.2500s±0.0087  | 0.1105s±0.0024            |
| vgg11              | 2.6793s±0.0036  | 1.1673s±0.0064            |
| vgg11_bn           | 3.5352s±0.0049  | 1.1585s±0.0071            |
| vgg13              | 4.0307s±0.0064  | 1.7766s±0.0094            |
| vgg13_bn           | 5.4520s±0.0038  | 1.7822s±0.0102            |
| vgg16              | 5.0354s±0.0185  | 2.1051s±0.0096            |
| vgg16_bn           | 6.5986s±0.0097  | 2.0979s±0.0030            |
| vgg19              | 6.0833s±0.0464  | 2.4351s±0.0064            |
| vgg19_bn           | 7.7534s±0.0083  | 2.4142s±0.0067            |
| resnet18           | 1.0552s±0.0136  | 0.6232s±0.0030            |
| resnet34           | 1.7585s±0.0118  | 0.9736s±0.0034            |
| resnet50           | 3.9019s±0.0311  | 2.4371s±0.0083            |
| resnet101          | 5.9299s±0.0081  | 3.6085s±0.0038            |
| resnet152          | 8.3714s±0.0160  | 5.0290s±0.0041            |
| squeezenet1_0      | 1.0074s±0.0051  | 0.5837s±0.0036            |
| squeezenet1_1      | 0.5603s±0.0020  | 0.3262s±0.0020            |
| densenet121        | 0.0000s±0.0000  | 0.0000s±0.0000            |
| densenet161        | 0.0000s±0.0000  | 0.0000s±0.0000            |
| densenet169        | 0.0000s±0.0000  | 0.0000s±0.0000            |
| densenet201        | 0.0000s±0.0000  | 0.0000s±0.0000            |
| inception_v3       | 1.8982s±0.0122  | 0.6891s±0.0050            |
| googlenet          | 1.4817s±0.0072  | 0.5958s±0.0043            |
| shufflenet_v2_x0_5 | 0.3818s±0.0009  | 0.3731s±0.0024            |
| shufflenet_v2_x1_0 | 0.4765s±0.0022  | 0.4678s±0.0018            |
| shufflenet_v2_x1_5 | 0.6756s±0.0041  | 0.6611s±0.0017            |
| shufflenet_v2_x2_0 | 1.1118s±0.0294  | 0.9726s±0.0053            |
| mobilenet_v2       | 2.6291s±0.0430  | 2.1853s±0.0384            |
| mobilenet_v3_small | 0.6796s±0.0028  | 0.6559s±0.0036            |
| mobilenet_v3_large | 1.7784s±0.0047  | 1.6223s±0.0323            |
| resnext50_32x4d    | 4.5525s±0.0040  | 3.4335s±0.0049            |
| resnext101_32x8d   | 11.1335s±0.0121 | 8.1321s±0.0115            |
| wide_resnet50_2    | 5.9996s±0.0168  | 3.0299s±0.0148            |
| wide_resnet101_2   | 10.0733s±0.0335 | 4.3592s±0.0220            |
| mnasnet0_5         | 1.1958s±0.0200  | 1.1879s±0.0119            |
| mnasnet0_75        | 1.9824s±0.0435  | 1.8158s±0.0118            |
| mnasnet1_0         | 2.2817s±0.0178  | 2.1115s±0.0235            |
| mnasnet1_3         | 3.4071s±0.0350  | 2.9455s±0.0616            |
<!-- benchmark ends -->

### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one

<!-- table starts -->
Update timestamp 08/06/2021 14:08:53

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
|      resnet18      | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|      resnet34      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|      resnet50      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     resnet101      | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|     resnet152      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|   squeezenet1_0    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|   squeezenet1_1    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    densenet121     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    densenet161     | :heavy_check_mark:  | :heavy_check_mark: |        :x:         |   False    |
|    densenet169     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    densenet201     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
| shufflenet_v2_x0_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
| shufflenet_v2_x1_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| shufflenet_v2_x1_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|    mobilenet_v2    | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
| mobilenet_v3_small | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
| mobilenet_v3_large | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|  resnext50_32x4d   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  wide_resnet50_2   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     mnasnet0_5     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|    mnasnet0_75     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |   False    |
|     mnasnet1_3     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
<!-- table ends -->