#include "kinectfusion.hpp"

__global__ void addKernel(Eigen::MatrixXf *a, Eigen::MatrixXf *b, Eigen::MatrixXf *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        (*c)(i) = (*a)(i) + (*b)(i);
    }
}

int test_cuda() {
    int N = 10;

    // Allocate host memory
    Eigen::MatrixXf h_a(N, 1), h_b(N, 1), h_c(N, 1);

    // Initialize matrices
    for (int i = 0; i < N; i++) {
        h_a(i) = static_cast<float>(i);
        h_b(i) = static_cast<float>(i * 2);
    }

    // Allocate device memory
    Eigen::MatrixXf *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, sizeof(Eigen::MatrixXf));
    cudaMalloc((void **)&d_b, sizeof(Eigen::MatrixXf));
    cudaMalloc((void **)&d_c, sizeof(Eigen::MatrixXf));

    // Copy data from host to device
    cudaMemcpy(d_a, &h_a, sizeof(Eigen::MatrixXf), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(Eigen::MatrixXf), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 4;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copy data back to host
    cudaMemcpy(&h_c, d_c, sizeof(Eigen::MatrixXf), cudaMemcpyDeviceToHost);

    // Print results
    std::cout << "Matrix A:" << std::endl;
    std::cout << h_a << std::endl;
    std::cout << "Matrix B:" << std::endl;
    std::cout << h_b << std::endl;
    std::cout << "Result:" << std::endl;
    std::cout << h_c << std::endl;

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
