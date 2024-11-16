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

// ReLU, Sigmoid function and their backward propagations are entry-wise independent
// Therefore they could be fully vectorized

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


class Tensor
{
    public:
        void cpu();
        void gpu();
        Tensor(std::vector<int>);
        ~Tensor();
        std::vector<float> h_dat; // store data on CPU
        float *d_dat;
        int size = 1; // initialize
    private:
        std::vector<int> shape;
        std::vector<int> stride;
        std::string device; // takes value 'cpu' or 'gpu'
};


Tensor::Tensor(std::vector<int> _shape){ // constructor
    stride = {1};
    shape = _shape;
    auto pos = stride.begin();
    for(const int n: shape){
        pos = stride.insert(pos, size); // calculate strides
        size *= n; // multiply all dimensions to get size
    }
    h_dat.resize(size);
    cudaMalloc((float**)&d_dat, size * 4); 
}


Tensor::~Tensor(){ // destructor
    cudaFree(d_dat);
}


void Tensor::cpu(){
    std::cout << "err = " << cudaMemcpy(h_dat.data(), d_dat, size * 4, cudaMemcpyDeviceToHost) << std::endl;
}

void Tensor::gpu(){
    std::cout << "err = " << cudaMemcpy(d_dat, h_dat.data(), size * 4, cudaMemcpyHostToDevice) << std::endl;
}


int main() {
    Tensor t({2, 3}), g({2, 3});
    t.h_dat = {-1, 2, 1, -2, 3, 4};
    t.gpu();

    sigmoid<<<CudaGetBlocks(t.size), kCudaThreadsNum>>>(t.d_dat, g.d_dat, t.size);

    g.cpu();
    std::cout << t.size << std::endl;
    std::cout << g.h_dat.size() << std::endl;
    for (int i = 0; i < g.size; ++i)
        std::cout << g.h_dat[i] << ' ';
    for (int i = 0; i < t.size; ++i)
        std::cout << t.h_dat[i] << ' ';
    return 0;
}

