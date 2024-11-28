cmake_minimum_required(VERSION 3.1)
project(C-2graph CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenMP REQUIRED)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
include_directories(${PROJECT_SOURCE_DIR})

add_executable(convert2binary utils/convert2binary.cpp)

add_executable(purn-ppr src/purn-ppr.cpp)

add_executable(purn-sssp src/purn-sssp.cpp)



find_package(CUDA REQUIRED)
if(${CUDA_FOUND})
    message("CUDA FOUND")
    set(CUDA_SOURCE_PROPERTY_FORMAT OBJ)
    set(CUDA_SEPARABLE_COMPILATION ON)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(CUDA_NVCC_FLAGS --expt-extended-lambda -Xcompiler -fopenmp -Xcompiler -pthread -lgomp -Xcompiler -mavx2 -arch=sm_86;-O3;-G;-g;-std=c++14) #事实证明-O能提高代码2x多的执行时间
    link_directories($ENV{CUDA_PATH}/lib/x64)
else(${CUDA_FOUND})
    MESSAGE(STATUS "cuda not found!")
endif(${CUDA_FOUND})


add_executable(sssp-base src/sssp/base/Main.cu)

add_executable(sssp-purn src/sssp/purn/Main.cu)

add_executable(ppr-base src/ppr/base/Main.cu)

add_executable(ppr-purn src/ppr/purn/Main.cu)