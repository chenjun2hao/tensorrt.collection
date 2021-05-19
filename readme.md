## 1. Updatas
- 2021.05.19 add [AttaNet AAAI2021](https://github.com/songqi-github/AttaNet) to benchmark
- 2021.04.06 add Efficientnet-b1，Efficientnetv2-m，and test on tx2
- 2021.03.31 add grid_sample plugin and sfnet semantic segmentation
- 2021.03.25 add rexnet and test on tx2,nx

## 2. Benchmark

|      category      |       model       |    mode    | size/(w,h) |  3080/ms  |    nx/ms     |    tx2/ms    |                                   url                                    |
| :----------------: | :---------------: | :--------: | :--------: | :-------: | :----------: | :----------: | :----------------------------------------------------------------------: |
| **Classification** |      rexnet       | float32/16 |  224*224   |  4.04/~   |    ~/8.9     |   ~/25.69    |               [rexnet](https://github.com/clovaai/rexnet)                |
|                    |  Efficientnet-b1  | float32/16 |  224*224   |  1.91/~   |      ~       | 17.47/15.62  |    [Efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch)     |
|                    |    Effnetv2_s     | float32/16 |  224*224   | 3.64/1.57 |      ~       | 32.02/24.99  |    [EfficientnetV2](https://github.com/d-li14/efficientnetv2.pytorch)    |
|                    |      rexnet       | float32/16 |  640*480   |     ~     |    ~/36.6    |   ~/79.27    |               [rexnet](https://github.com/clovaai/rexnet)                |
|  **Segmentation**  |       sfnet       | float32/16 |  640*480   | 8.79/2.71 | 109.74/50.03 | 150.87/99.57 |               [sfnet](https://github.com/lxtGH/SFSegNets)                |
|                    |   hrnetw18_ocr    | float32/16 |  640*480   |     ~     |   ~/65.565   |   ~/183.81   |    [hrnet_ocr](https://github.com/HRNet/HRNet-Semantic-Segmentation)     |
|                    | ddrnet23_slim_ocr | float32/16 |  640*480   |     ~     |   ~/17.805   |   ~/47.41    |              [ddrnet](https://github.com/ydhongHIT/DDRNet)               |
|                    |   ddrnet23_ocr    | float32/16 |  640*480   |     ~     |     ~/23     |     ~/93     |              [ddrnet](https://github.com/ydhongHIT/DDRNet)               |
|                    |    lite_hrnet     | float32/16 |  640*480   | 6.23/5.05 |      ~       |      ~       |            [lite_hrnet](https://github.com/HRNet/Lite-HRNet)             |
|                    |     mobilev2      | float32/16 |  640*480   |  ~/3.14   |   ~/62.01    |   ~/137.85   | [mobilev2](https://github.com/CSAILVision/semantic-segmentation-pytorch) |
|                    |   resnet50_psp    | float32/16 |  640*480   |  ~/6.56   |   ~/148.02   |   ~/422.23   | [resnet50](https://github.com/CSAILVision/semantic-segmentation-pytorch) |
|                    |      AttaNet      | float32/16 |  640*480   |    ~/~    |     ~/~      | 61.45/43.33  |        [atta-resnet18](https://github.com/songqi-github/AttaNet)         |

## 3. 编译

3.1 requirements
- [tensorrt 7.x and plugin](https://github.com/chenjun2hao/TensorRT/tree/release/7.2)

3.2 编译
修改根目录下的`CMakeLists.txt`，其中
- TENSORRT_ROOT：是tensorrt解压文件所在的位置

然后直接编译
```
mkdir build && cd build
cmake ..
make -j4
```

## 4. 测试
### 4.1 分类验证测试
将tensorrt的分类结果和pytorch的输出数值做对比,测试用了7张imagenet中的7张图片，在`images/`文件夹下。

```
./eval path/model.trt
```
输出例子，比如：
```
cost time is:8.34ms  |  predict: 309 9.45473
cost time is:4.05ms  |  predict: 309 9.56318
cost time is:4.03ms  |  predict: 599 9.31222
cost time is:4.03ms  |  predict: 304 6.2007
cost time is:4.04ms  |  predict: 310 12.2731
cost time is:4.03ms  |  predict: 327 8.02724
cost time is:4.05ms  |  predict: 310 8.6457
```

### 4.2 分类模型测试

图像分类模型的推理脚本

```
./classify path_to_model/model.trt path_to_image/image.jpg
```

### 4.3 语义分割模型测试

语义分割模型的推理，可视化脚本，基于`cityscapes`数据集做的推理脚本例子

```bash
./seg path_to_model/model.trt path_to_image/image.jpg
```

## other
- 在做tensorrt模型转换和推理的时候，有可能需要一些插件，请参考:[tensorrt](https://github.com/chenjun2hao/TensorRT/tree/release/7.2)