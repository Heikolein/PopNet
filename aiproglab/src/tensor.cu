#include "cuda_config.cuh"
#include "tensor.cuh"

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
    std::cout << cudaMemcpy(h_dat.data(), d_dat, size * 4, cudaMemcpyDeviceToHost) << std::endl;
}

void Tensor::gpu(){
    std::cout << cudaMemcpy(d_dat, h_dat.data(), size * 4, cudaMemcpyHostToDevice) << std::endl;
}
