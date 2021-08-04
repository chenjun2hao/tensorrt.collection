cmake_minimum_required(VERSION 2.8)
project(inference)

find_package(catkin_simple REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CUDA_INCLUDE_DIRS}
)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast ")
set(CUDA_NVCC_FLAGS  "-D_FORCE_INLINES -Xcompiler -fPIC ${GENCODES}")


file(GLOB INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/../include/common/logger.cpp)

set(srcs ${INFER_SRC})
cuda_add_library(segmentation_api SHARED ${srcs})      # 会自动链接cuda的动态库
target_link_libraries(segmentation_api
                      nvinfer                       # 不用包含tensorrt plugin层的头文件，采用原有的头文件就可以
                      nvinfer_plugin
                    )


# 2. 安装
cs_install()
cs_export(INCLUDE_DIRS include ${CMAKE_CURRENT_BINARY_DIR})