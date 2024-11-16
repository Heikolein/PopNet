#ifndef TENSOR_H_
#define TENSOR_H_

#include "cuda_config.cuh"
#include <vector>
#include <string>

class Tensor
{
    public:
        void cpu();
        void gpu();
        Tensor(std::vector<int> _shape);
        ~Tensor();
        std::vector<float> h_dat; // store data on CPU
        float *d_dat;
        int size = 1; // initialize
    private:
        std::vector<int> shape;
        std::vector<int> stride;
        std::string device; // takes value 'cpu' or 'gpu'
};

#endif