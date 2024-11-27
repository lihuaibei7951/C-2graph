cmake_minimum_required(VERSION 3.1)
project(C-2graph CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR})

add_executable(convert2binaryweight utils/convert2binaryweight.cpp)

add_executable(purn-ppr src/purn-ppr.cpp)

add_executable(purn-sssp src/purn-sssp.cpp)

