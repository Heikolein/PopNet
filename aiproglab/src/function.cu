#include "cuda_config.cuh"

__global__ void relu(float* in, float* out, int n) {
    CUDA_KERNEL_LOOP(i, n){
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}


__global__ void sigmoid(float* in, float* out, int n){
    CUDA_KERNEL_LOOP(i, n){
        out[i] = 1 / (1 + expf(- in[i]));
    }
}


__global__ void relu_backward(float* grad, float* in, float* out, int n) {
    CUDA_KERNEL_LOOP(i, n){
        out[i] = in[i] > 0 ? grad[i] : 0;
    }
}


__global__ void sigmoid_backward(float* grad, float* y, float* out, int n){
    CUDA_KERNEL_LOOP(i, n){
        out[i] = grad[i] * y[i] * (1 - y[i]);
    }
}