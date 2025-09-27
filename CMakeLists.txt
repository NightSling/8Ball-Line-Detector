cmake_minimum_required(VERSION 3.16)
project(ThinExtendedLines)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(ThinExtendedLines main.cpp)

# Link OpenCV libraries
target_link_libraries(ThinExtendedLines ${OpenCV_LIBS})
