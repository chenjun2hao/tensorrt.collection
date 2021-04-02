## 1. Updatas
- 2020.03.25 add rexnet and test on tx2,nx
- 2020.03.31 add grid_sample plugin and sfnet semantic segmentation

## 2. Benchmark

|      category      |       model       | batchsize |     mode      | size/(w,h) |  3080/ms  |    nx/ms     |    tx2/ms    |                                url                                |
| :----------------: | :---------------: | :-------: | :-----------: | :--------: | :-------: | :----------: | :----------: | :---------------------------------------------------------------: |
| **Classification** |      rexnet       |     1     |    float16    |  224*224   |           |     8.9      |    25.69     |            [rexnet](https://github.com/clovaai/rexnet)            |
|                    |      rexnet       |     1     |    float16    |  640*480   |           |     36.6     |    79.27     |            [rexnet](https://github.com/clovaai/rexnet)            |
|  **Segmentation**  |       sfnet       |     1     | int32/float16 |  640*480   | 8.79/2.71 | 109.74/50.03 | 150.87/99.57 |            [sfnet](https://github.com/lxtGH/SFSegNets)            |
|                    |   hrnetw18_ocr    |     1     | int32/float16 |  640*480   |     ~     |   ~/65.565   |   ~/183.81   | [hrnet_ocr](https://github.com/HRNet/HRNet-Semantic-Segmentation) |
|                    | ddrnet23_slim_ocr |     1     | int32/float16 |  640*480   |     ~     |   ~/17.805   |   ~/47.41    |           [ddrnet](https://github.com/ydhongHIT/DDRNet)           |


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

## other
- 在做tensorrt模型转换和推理的时候，有可能需要一些插件，请参考:[tensorrt](https://github.com/chenjun2hao/TensorRT/tree/release/7.2)