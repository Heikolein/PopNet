#include <cublas_v2.h>
#include "linalg.cuh"
#include "cuda_config.cuh"

void fullyConnectedForward(float* input, float* weights, float* bias, float* output, const int batch_size, const int in_features, const int out_features){
    void* pointers[] = {(float*)input, (float*)weights, (float*) bias, (float*) output};
    check_ptr_on_device(pointers, 4);
    // check whether size of the matrices match (to be added)

    gemm_gpu<<<CudaGetBlocks(batch_size * out_features), kCudaThreadsNum>>>(weights, input, bias, output, 1, batch_size, in_features, out_features);
    checkCudaError(cudaGetLastError());
}

void fullyConnectedBackward(float* input, float* weights, float *bias, float* output, \
                            float* grad_output, float* grad_input, float* grad_weight, float* grad_bias, \
                            const int batch_size, const int in_features, const int out_features) {
    void* pointers[] = {(float*)input, (float*)weights, (float*) bias, (float*) output, \
                        (float*)grad_input, (float*)grad_weights, (float*)grad_bias, (float*)grad_output};
    check_ptr_on_device(pointers, 8);
    // check whether size of the matrices match (to be added)

    gemm_gpu<<<CudaGetBlocks()>>>()
}