#ifndef FUNCTION_H_
#define FUNCTION_H_

#include "cuda_config.cuh"

__global__ void relu(float* in, float* out, int n);
__global__ void sigmoid(float* in, float* out, int n);
__global__ void relu_backward(float* grad, float* in, float* out, int n);
__global__ void sigmoid_backward(float* grad, float* y, float* out, int n);

#endif