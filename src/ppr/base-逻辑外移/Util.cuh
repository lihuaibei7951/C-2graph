#include <iostream>
#include <vector>
#include <string> //otherwise, you cannot use "+" for two strings
#include <algorithm>
#include "cub/cub.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h" //making threadIdx/blockIdx/blockDim/gridDim visible

#define FULL_MASK 0xffffffff
//typedef float ValueType;

using namespace std;

static void HandleError(cudaError_t err, const char *file, int line){
	if (err != cudaSuccess){
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(-1);
	}
}

#define CUDA_ERROR( err ) (HandleError( err, __FILE__, __LINE__))

#define ASSERT(flag) {if (!(flag)) {cerr<<"ASSERT FAIL @ "<<__FILE__<<":"<<__LINE__<<endl; exit(1);}}

#define CALC_BLOCKS_NUM(ITEMS_PER_BLOCK, CALC_SIZE) MAX_BLOCKS_NUM<((CALC_SIZE-1)/ITEMS_PER_BLOCK+1)? MAX_BLOCKS_NUM:((CALC_SIZE-1)/ITEMS_PER_BLOCK+1)

//GPU configuration
const static size_t THREADS_PER_BLOCK = 1024; //maximum is 512
const static size_t THREADS_PER_WARP = 32; //warp size
//const static size_t MAX_BLOCKS_NUM = 68 * (1536/THREADS_PER_BLOCK); // 80MP, 2048 threads/MP
//const static size_t MAX_BLOCKS_NUM = 80 * (2048/THREADS_PER_BLOCK); // 80MP, 2048 threads/MP
const static size_t MAX_BLOCKS_NUM = 68;
const static size_t MAX_THREADS_NUM = MAX_BLOCKS_NUM * THREADS_PER_BLOCK;

