cmake_minimum_required(VERSION 3.2)
project(segmentation_catkin)

find_package(catkin REQUIRED) 
find_package(CUDA REQUIRED)

# 创建ros package的.cmake + 设置 libs out 到 devel 下
catkin_package(
  INCLUDE_DIRS
    include               
  LIBRARIES
    ${PROJECT_NAME}
)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE release)

# set the library out path
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CATKIN_DEVEL_PREFIX}/lib)

# 0. 根据 cuda 的版本判断 gpu 算力
# set nvidia card 30x->86, tx2->62, nx->62
# set(GPU_ARCHS 86)
if (DEFINED GPU_ARCHS)
  message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
  separate_arguments(GPU_ARCHS)
else()
  list(APPEND GPU_ARCHS
      62
    )
endif()

if (CUDA_VERSION VERSION_GREATER_EQUAL 11.1)
  list(APPEND GPU_ARCHS 86)
else()
  message(WARNING "Detected CUDA version is < 11.1. SM86 not supported.")
endif()

foreach(arch ${GPU_ARCHS})
    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
endforeach()

# 1. 编译成动态库
# 本机上需要指定 tensorrt 的路径，tx2/nx上并不需要
set(TENSORRT_ROOT "/home/darknet/CM/profile/TensorRT-7.2.2.3")
include_directories(${TENSORRT_ROOT}/include)
link_directories(${TENSORRT_ROOT}/lib)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDA_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast ")
set(CUDA_NVCC_FLAGS  "-D_FORCE_INLINES -Xcompiler -fPIC ${GENCODES}")


file(GLOB INFER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
                    ${CMAKE_CURRENT_SOURCE_DIR}/include/common/logger.cpp)

set(srcs ${INFER_SRC})
cuda_add_library(${PROJECT_NAME} SHARED ${srcs})      # cuda_add_library 会自动链接cuda的动态库
target_link_libraries(${PROJECT_NAME}
                      nvinfer                         # 不用包含tensorrt plugin层的头文件，采用原有的头文件就可以
                      nvinfer_plugin
                    )

# 1. 例子：如何使用 segmentation 动态库创建一个 exe （可选）
# find_package(OpenCV REQUIRED)

# include_directories(
#     ${OpenCV_INCLUDE_DIRS}
#     ./include
# )
# add_executable(seg segmentation/seg.cpp)
# target_link_libraries(seg 
#                     ${PROJECT_NAME}
#                     ${OpenCV_LIBS})  
# install(TARGETS seg 
#   LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
#   RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
# )

# 2. 安装
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
install(TARGETS ${PROJECT_NAME} 
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)


