# 语义分割部分打包成 catkin 的包

## 1. 整体流程

1. 首先语义分割的核心部分需要能编译成动态库
2. 然后将整个项目打包成 catkin 的包


## 2. 能编译成动态库
就按通用的 cmake 项目的书写方式，能用 cmake 的 add_library 编译成动态库就行

## 3. 将整个项目打包成 catkin 的包
打包成 catkin 的包主要是为了，用`package.xml`进行 depend 管理。（**但是这种情况还是容易出现动态库先后依赖的问题，依赖的库还没有编译出来**）

1. 新建一个`package.xml`和`CMakeLists.txt`。


2. `package.xml`的 name 和文件夹的名字， cmakelists.txt中 project_name, 动态库的名字， 这四者的名字需要一样。


3. 在cmakelists.txt中导出`<Package>Config.cmake`文件，
```c
# 1. 采用catkin的方式
find_package(catkin REQUIRED) 

# 创建ros package的.cmake + 设置 libs out 到 devel 下
catkin_package(     # 设定头文件和动态库
  INCLUDE_DIRS
    include         # 并不是一个路径，表示当前路径下的include文件夹
  LIBRARIES
    ${PROJECT_NAME} # 动态库的名字
)

# 2. 采用 catkin_simple 的方式
...
```
**同时，catkin_package 这条命令会把编译的动态库放到`devel/lib`下**
参考的`<Package>Config.cmake`在`devel/share/<catkin_package_name>/cmake`下，比如：devel/share/segmentation_catkin/cmake


## 4. 测试
新建另外一个`test` catkin 包，注意4个名字相同，这里去掉一个动态库，那就是3个名字。

然后在`package.xml`中添加：
```
<depend>segmentation_catkin</depend>
```

在cmakelists.txt中可以打印一些变量
```c
cmake_minimum_required(VERSION 3.2)
project(test_segmentation)

find_package(catkin_simple REQUIRED)
catkin_simple()

message("111111111", ${segmentation_catkin_FOUND})
message("333333333", ${eigen_catkin_FOUND})
message("444444444", ${hdmap_karto_FOUND})
message("222222222", ${catkin_LIBRARIES})
```

```
find_package(catkin_simple REQUIRED)
catkin_simple()
```
采用上面两句去解析`package.xml`

常用的调试变量：
- <catkin_package>_FOUND: catkin_simple是否找到了某catkin包
- catkin_LIBRARIES：如果找到了某catkin包，他的动态库会添加到这个
- catkin_INCLUDE_DIRS:如果找到了某catkin包，他的头文件路径会添加到这个