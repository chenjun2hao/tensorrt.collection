## 1. Updatas
- 2021.07.08 add [STDC-Seg 2021](https://github.com/MichaelFan01/STDC-Seg) to benchmark
- 2021.07.05 add [SegFormer 2021](https://github.com/NVlabs/SegFormer) to benchmark
- 2021.05.19 add [AttaNet AAAI2021](https://github.com/songqi-github/AttaNet) to benchmark
- 2021.04.06 add Efficientnet-b1，Efficientnetv2-m，and test on tx2
- 2021.03.31 add grid_sample plugin and sfnet semantic segmentation
- 2021.03.25 add rexnet and test on tx2,nx

## 2. Benchmark

**2.1 Classification**

|      model      | mode  | size/(w,h) |  params   | Flops | 3080/ms | nx/ms  |   tx2/ms    |                                url                                 |
| :-------------: | :---: | :--------: | :-------: | :---: | :-----: | :----: | :---------: | :----------------------------------------------------------------: |
|     rexnet      | 32/16 |  224*224   |  4.04/~   |       |         | ~/8.9  |   ~/25.69   |            [rexnet](https://github.com/clovaai/rexnet)             |
| Efficientnet-b1 | 32/16 |  224*224   |  1.91/~   |       |         |   ~    | 17.47/15.62 | [Efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch)  |
|   Effnetv2_s    | 32/16 |  224*224   | 3.64/1.57 |       |         |   ~    | 32.02/24.99 | [EfficientnetV2](https://github.com/d-li14/efficientnetv2.pytorch) |
|     rexnet      | 32/16 |  640*480   |     ~     |       |         | ~/36.6 |   ~/79.27   |            [rexnet](https://github.com/clovaai/rexnet)             |

**2.2 Segmentation**
|         model         | mode  | size/(w,h) | params/M | Flops/G |  3080/ms  |    nx/ms     |    tx2/ms     |                                   url                                    |
| :-------------------: | :---: | :--------: | :------: | :-----: | :-------: | :----------: | :-----------: | :----------------------------------------------------------------------: |
|         sfnet         | 32/16 |  640*480   |          |         | 8.79/2.71 | 109.74/50.03 | 150.87/99.57  |               [sfnet](https://github.com/lxtGH/SFSegNets)                |
|    hrnetv2_w18_ocr    | 32/16 |  640*480   |          |         |     ~     |   ~/65.565   |   ~/183.81    |    [hrnet_ocr](https://github.com/HRNet/HRNet-Semantic-Segmentation)     |
| hrnetv2_w18_ocr_ssld  | 32/16 |  640*480   |          |         |     ~     |     ~/~      |     ~/160     |    [hrnet_ocr](https://github.com/HRNet/HRNet-Semantic-Segmentation)     |
| hrnetv2_w18_ssld_P0.1 | 32/16 |  640*480   |          |         |     ~     |     ~/~      |     ~/77      |    [hrnet_ocr](https://github.com/HRNet/HRNet-Semantic-Segmentation)     |
|   ddrnet23_slim_ocr   | 32/16 |  640*480   |          |         |     ~     |   ~/17.805   |  55.03/38.89  |              [ddrnet](https://github.com/ydhongHIT/DDRNet)               |
|     ddrnet23_ocr      | 32/16 |  640*480   |          |         |     ~     |     ~/23     |     ~/93      |              [ddrnet](https://github.com/ydhongHIT/DDRNet)               |
|      lite_hrnet       | 32/16 |  640*480   |          |         | 6.23/5.05 |      ~       |       ~       |            [lite_hrnet](https://github.com/HRNet/Lite-HRNet)             |
|       mobilev2        | 32/16 |  640*480   |          |         |  ~/3.14   |   ~/62.01    |   ~/137.85    | [mobilev2](https://github.com/CSAILVision/semantic-segmentation-pytorch) |
|     resnet50_psp      | 32/16 |  640*480   |          |         |  ~/6.56   |   ~/148.02   |   ~/422.23    | [resnet50](https://github.com/CSAILVision/semantic-segmentation-pytorch) |
|        AttaNet        | 32/16 |  640*480   |          |         |    ~/~    |     ~/~      |  61.45/43.33  |        [atta-resnet18](https://github.com/songqi-github/AttaNet)         |
|       SegFormer       | 32/16 |  640*480   |          |         |    ~/~    |     ~/~      | 633.02/521.88 |          [SegFormer 2021](https://github.com/NVlabs/SegFormer)           |
|       STDC-Seg        | 32/16 |  640*480   |  14.186  | 25.327  |    ~/~    |     ~/~      |  60.57/47.23  |        [STDC-Seg 2021](https://github.com/MichaelFan01/STDC-Seg)         |



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