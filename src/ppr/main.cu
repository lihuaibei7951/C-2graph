#include <iostream>
#include <chrono>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void waitKernel() {
    // do nothing
}

void waitTime(int seconds) {
    auto startTime = std::chrono::high_resolution_clock::now();
    auto targetTime = startTime + std::chrono::seconds(seconds);
    do {
        waitKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    } while (std::chrono::high_resolution_clock::now() < targetTime);
}

int main() {
    // 申请6GB的显存
    int* degree;
    CUDA_CHECK(cudaMalloc(&degree, sizeof(int) * 1024*1024*1024));

    // 等待一段时间（100秒）
    waitTime(10000);

    // 释放显存
    CUDA_CHECK(cudaFree(degree));

    // 结束进程
    exit(EXIT_SUCCESS);
}