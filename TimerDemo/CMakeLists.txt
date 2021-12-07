# cmake needs this line
cmake_minimum_required(VERSION 3.1)
 
# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Define project name
project(timer_demo)

file(GLOB CURRENT_HEADERS  *.h)
file(GLOB CURRENT_SOURCES  *.c)

add_executable(${PROJECT_NAME} ${CURRENT_HEADERS} ${CURRENT_SOURCES})
target_link_libraries(${PROJECT_NAME} pthread)
