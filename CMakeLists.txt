cmake_minimum_required(VERSION 3.20)

project(fast-inference
    VERSION 0.1
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(fast-inference
    src/tensor.cpp
    src/operator.cpp
)

target_include_directories(fast-inference
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)