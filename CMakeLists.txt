cmake_minimum_required(VERSION 3.15)
project(testEigen LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

include_directories("./ext/eigen")


add_executable(testEigen main.cpp)

add_executable(testEigenGPU main_gpu.cu)
add_compile_definitions(EIGEN_DONT_VECTORIZE=1)
