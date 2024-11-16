#include <cublas_v2.h>
#include <cuda_runtime.h>

void check_ptr_on_device(void** ptrs, int num) {
    for (int i = 0; i < num; i++) {
        cudaPointerAttributes attributes;
        cudaError_t error = cudaPointerGetAttributes(&attributes, ptrs[i]);
        
        if (error == cudaSuccess) {
            if (attributes.memoryType == cudaMemoryTypeHost) {
                throw std::runtime_error("Data not on GPU");
            }
        } else {
            std::cerr << "Failed to get pointer attributes for pointer " << i << ": " << cudaGetErrorString(error) << std::endl;
            throw std::runtime_error("");
        }
    }
}

// X(m*n) = A(m*k) * B(k*n) + C(m*n)
__device__ void gemm_gpu(const float *A, const float *B, float *C, float *X, const float beta, const int m, const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float *alpha = &alf;
    // Create a handle for CUBLAS
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed" << std::endl;
        return EXIT_FAILURE;
    }
    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, B, n, A, k, beta, C, n);
    cudaDeviceSynchronize();
    cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, ldc, 1, C, 0, C, X, n);
    cudaDeviceSynchronize();
    // Destroy the handle
    cublasDestroy(handle);
}
