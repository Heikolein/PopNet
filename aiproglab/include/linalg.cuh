#ifndef LINALG_H_
#define LINALG_H_

#include "cuda_config.cuh"

void check_ptr_on_device(void** ptrs, int num);
__device__ void gemm_gpu(const float *A, const float *B, float *C, float *X, const float beta, const int m, const int k, const int n);

#endif