# cmake needs this line
cmake_minimum_required(VERSION 3.1)
 
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(cv_read_save)
 
find_package(OpenCV REQUIRED) 

file(GLOB CURRENT_HEADERS  *.h)
file(GLOB CURRENT_SOURCES  *.cpp)

add_executable(${PROJECT_NAME} ${CURRENT_HEADERS} ${CURRENT_SOURCES})

target_link_libraries(${PROJECT_NAME} opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)
