#include <curand.h>

// Fill the matrix with random numbers on GPU
void matrix_init(float *A, int rows, int cols) {
    // Create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    // Set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long)clock());
    // Fill the array with random numbers on the device
    curandGenerateUniform(prng, A, rows * cols);
    curandDestroyGenerator(prng);
}
