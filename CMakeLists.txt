cmake_minimum_required(VERSION 3.2)
project(tensorrt_template)

# set lib, exe output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE release)
# set nvidia card 30x->86, tx2->62, nx->62
set(GPU_ARCHS 86)
set(TENSORRT_ROOT /home/darknet/CM/profile/TensorRT-7.2.2.3)
# set(Plugin_lib /home/darknet/CM/12_tensorrt/TensorRT/libs)

include_directories(${TENSORRT_ROOT}/include)

link_directories(${TENSORRT_ROOT}/lib)

add_subdirectory(src)
add_subdirectory(rexnet)
add_subdirectory(segmentation)
add_subdirectory(module)

