cmake_minimum_required(VERSION 3.1)
project(C-2graph CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

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

# 添加 CUDA 和 C++ 源文件
set(SOURCE_FILES1
        src/sssp/base-perk/SSSP.cpp
        src/sssp/base-perk/Main.cu
)

set(SOURCE_FILES2
        src/sssp/purn-perk/SSSP.cpp
        src/sssp/purn-perk/Main.cu
)

# 创建可执行文件
add_executable(sssp-base-M ${SOURCE_FILES1})
add_executable(sssp-purn-M ${SOURCE_FILES2})  # 假设有多个可执行文件

# 设置编译选项
foreach(target sssp-base-M sssp-purn-M)  # 对每个目标应用编译选项
    target_compile_options(${target} PRIVATE -std=c++14)  # 确保使用 C++14 标准
endforeach()

# 设置链接选项
foreach(target sssp-base-M sssp-purn-M)  # 对每个目标应用链接选项
    target_link_libraries(${target} PRIVATE
            pthread
            ${CUDA_LIBRARIES}  # 包含 cudart 库
    )
endforeach()

# 指定 CMake 使用 CUDA 编译器进行 CUDA 文件编译
foreach(target sssp-base-M sssp-purn-M)  # 对每个目标应用目标属性
    set_target_properties(${target} PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON  # 启用可分离编译（如果需要的话）
    )
endforeach()




