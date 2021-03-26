## 1. Updatas
- 2020.03.25 add rexnet and test on tx2,nx

## 2. Benchmark

| model  | batchsize |  mode   | size/(w,h) | nx/ms | tx2/ms |                     url                     |
| :----: | :-------: | :-----: | :--------: | :---: | :----: | :-----------------------------------------: |
| rexnet |     1     | float16 |  224*224   |  8.9  |   ~    | [rexnet](https://github.com/clovaai/rexnet) |
| rexnet |     1     | float16 |  640*480   | 36.6  |   ~    | [rexnet](https://github.com/clovaai/rexnet) |


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