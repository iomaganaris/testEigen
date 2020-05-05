cmake_minimum_required(VERSION 3.15)
project(testEigen)
set(CMAKE_CXX_STANDARD 14)

include_directories("./ext/eigen")


add_executable(testEigen main.cpp)
