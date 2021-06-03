# simplify

[![tests](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml/badge.svg)](https://github.com/EIDOSlab/simplify/actions/workflows/test.yaml)

| Architecture   | Pruned time    | Simplified time (p=50%)   |
|----------------|----------------|---------------------------|
| alexnet        | 0.2457s±0.0066 | 0.1077s±0.0020            |
| resnet18       | 1.0458s±0.0073 | 0.6201s±0.0018            |
| resnet34       | 1.7364s±0.0110 | 0.9600s±0.0011            |
| resnet50       | 3.8327s±0.1174 | 2.4317s±0.0051            |
| resnet101      | 5.9290s±0.0298 | 3.6026s±0.0020            |
| resnet152      | 8.4356s±0.0046 | 5.0217s±0.0036            |
| squeezenet1_0  | 1.0077s±0.0021 | 0.5853s±0.0053            |
| squeezenet1_1  | 0.5596s±0.0045 | 0.3235s±0.0012            |
| vgg16          | 4.9776s±0.0145 | 2.0713s±0.0046            |
| vgg16_bn       | 6.5032s±0.0037 | 2.0776s±0.0073            |
| vgg19          | 6.0119s±0.0489 | 2.3946s±0.0012            |
| vgg19_bn       | 7.6872s±0.0080 | 2.3907s±0.0009            |

### Status of torchvision.models

:heavy_check_mark:: all good

:x:: gives different results

:cursing_face:: an exception occurred

:man_shrugging:: test skipped due to failing of the previous one

<!-- table starts -->
Update timestamp 03/06/2021 16:00:30

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
|    densenet121     |   :cursing_face:    |  :man_shrugging:   |  :man_shrugging:   |   False    |
|    densenet161     |   :cursing_face:    |  :man_shrugging:   |  :man_shrugging:   |   False    |
|    densenet169     |   :cursing_face:    |  :man_shrugging:   |  :man_shrugging:   |   False    |
|    densenet201     |   :cursing_face:    |  :man_shrugging:   |  :man_shrugging:   |   False    |
|    inception_v3    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     googlenet      | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
| shufflenet_v2_x0_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |    True    |
| shufflenet_v2_x1_0 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |    True    |
| shufflenet_v2_x1_5 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |    True    |
| shufflenet_v2_x2_0 | :heavy_check_mark:  | :heavy_check_mark: |   :cursing_face:   |    True    |
|    mobilenet_v2    | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
| mobilenet_v3_small | :heavy_check_mark:  |        :x:         |  :man_shrugging:   |    True    |
| mobilenet_v3_large | :heavy_check_mark:  |        :x:         |  :man_shrugging:   |    True    |
|  resnext50_32x4d   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
|  resnext101_32x8d  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
|  wide_resnet50_2   | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|  wide_resnet101_2  | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |   False    |
|     mnasnet0_5     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
|    mnasnet0_75     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
|     mnasnet1_0     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
|     mnasnet1_3     | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |    True    |
<!-- table ends -->