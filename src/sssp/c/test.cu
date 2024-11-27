#include <stdio.h>

// 定义向量长度
#define N 10

// 核函数：向量相加
__global__ void vectorAddKernel(int *a, int *b, int *c) {
    // 获取当前线程的全局索引
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 确保索引在向量长度范围内
    
    int iter = 0;
    while(iter<1){
    iter++;
    int x;
	if (tid < N) {
        // 将相应位置的元素相加，并将结果存储在 c 中
        x = b[tid];
        c[tid] = x;
    }

    }
    
}

int main(int argc, char **argv) {
    // 定义向量 a, b, c
    int a[N], b[N], c[N];
    int *dev_a, *dev_b, *dev_c;

    // 在设备上分配内存
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));

    // 初始化向量 a, b
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // 将向量 a, b 复制到设备
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

    // 定义块大小和网格大小
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // 调用核函数
    vectorAddKernel<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c);

    // 将结果复制回主机
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Vector addition result:\n");
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // 释放设备上的内存
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}