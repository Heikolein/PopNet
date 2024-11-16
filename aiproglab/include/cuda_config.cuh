#ifndef CUDA_CONFIG_H
#define CUDA_CONFIG_H

#include <math.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return(N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for(int i = blockIdx.x * blockDim.x + threadIdx.x; i<(n); \
  i += blockDim.x * gridDim.x)

#endif