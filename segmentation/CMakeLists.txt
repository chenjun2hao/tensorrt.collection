cmake_minimum_required(VERSION 2.8)
project(segmentation)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ../include
)

add_executable(seg seg.cpp)
target_link_libraries(seg 
                    inference_api
                    ${OpenCV_LIBS})      
 



