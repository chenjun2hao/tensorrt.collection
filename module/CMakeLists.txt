cmake_minimum_required(VERSION 2.8)
project(test)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ../include
)

link_directories(${TENSORRT_LIB_PATH})

add_executable(template template.cpp)
target_link_libraries(template 
                    inference_api
                    ${OpenCV_LIBS})      
 



