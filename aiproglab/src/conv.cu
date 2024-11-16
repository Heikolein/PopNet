#include "cuda_config.cuh"

__global__ void im2col_kernel(float* input, float* output, const int batch_size, const int kernel_size, const int in_channel, \
                                const int height, const int width){
    int index = blockDim.x * blockIdx.x + threadIdx.x
    int num = in_channel * height * width;
    

}

__global__ void col2im_kernel(float* input, float* output, const int batch_size, const int kernel_size, const int in_channel, \
                                 const int height, const int width){
    
}

__global__ void conv2d_forward(float* input, float* bias, float* weight, float* output, const int batch_size, const int kernel_size,
                                int in_channel){
    cublasHandle_t handle;
    cublasCreate(&handle);
    int col_weight = out_channel * 
}

__global__ void conv2d_backward(){

}

