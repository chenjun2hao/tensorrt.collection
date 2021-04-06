## 1. Updatas
- 2020.03.25 add rexnet and test on tx2,nx
- 2020.03.31 add grid_sample plugin and sfnet semantic segmentation

## 2. Benchmark

|      category      |       model       |    mode    | size/(w,h) |  3080/ms  |    nx/ms     |    tx2/ms    |                                url                                |
| :----------------: | :---------------: | :--------: | :--------: | :-------: | :----------: | :----------: | :---------------------------------------------------------------: |
| **Classification** |      rexnet       | float32/16 |  224*224   |  4.04/~   |    ~/8.9     |   ~/25.69    |            [rexnet](https://github.com/clovaai/rexnet)            |
|                    |  Efficientnet-b1  | float32/16 |  224*224   |  1.91/~   |      ~       |      ~       | [Efficientnet](https://github.com/lukemelas/EfficientNet-PyTorch) |
|                    |      rexnet       | float32/16 |  640*480   |           |    ~/36.6    |   ~/79.27    |            [rexnet](https://github.com/clovaai/rexnet)            |
|  **Segmentation**  |       sfnet       | float32/16 |  640*480   | 8.79/2.71 | 109.74/50.03 | 150.87/99.57 |            [sfnet](https://github.com/lxtGH/SFSegNets)            |
|                    |   hrnetw18_ocr    | float32/16 |  640*480   |     ~     |   ~/65.565   |   ~/183.81   | [hrnet_ocr](https://github.com/HRNet/HRNet-Semantic-Segmentation) |
|                    | ddrnet23_slim_ocr | float32/16 |  640*480   |     ~     |   ~/17.805   |   ~/47.41    |           [ddrnet](https://github.com/ydhongHIT/DDRNet)           |


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