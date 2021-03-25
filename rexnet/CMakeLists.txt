cmake_minimum_required(VERSION 2.8)
project(demo)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ../include
)

link_directories(${TENSORRT_LIB_PATH})

add_executable(classify demo.cpp)
target_link_libraries(classify 
                    inference_api
                    ${OpenCV_LIBS})      
 



