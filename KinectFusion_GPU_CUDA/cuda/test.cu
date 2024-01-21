#include "kinectfusion.hpp"

__global__ void addKernel(float *a, float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}

int test_cuda() {
    int N = 10;

    // Allocate host memory and initialize matrices
    Eigen::VectorXf vec_a = Eigen::VectorXf::LinSpaced(N, 0, N-1);
    Eigen::VectorXf vec_b = 2 * vec_a;

    Eigen::MatrixXf h_a = vec_a;
    Eigen::MatrixXf h_b = vec_b;
    Eigen::MatrixXf h_c(N, 1);

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 4;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copy data back to host
    cudaMemcpy(h_c.data(), d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Matrix A:" << std::endl << h_a << std::endl;
    std::cout << "Matrix B:" << std::endl << h_b << std::endl;
    std::cout << "Result:" << std::endl << h_c << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
