cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 14)

project(gmm_sort)

include_directories(
    ${PROJECT_SOURCE_DIR}/3rdparty/opencv4
)

set(OpenCV_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cmake)
find_package(OpenCV REQUIRED)
message("OpenCV_LIBS: " ${OpenCV_LIBS})
add_executable(gmm_sort gmm_sort.cpp)

target_link_libraries(
     gmm_sort
     ${OpenCV_LIBS}
 )
