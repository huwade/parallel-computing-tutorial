// CUDA runtime
#include <cuda_runtime.h>
#include "matmul.h"
#include <iostream>
#define BLOCK_SIZE 32

/**
 * Reference: Kernel 2: Global Memory Coalescing
 * Link : https://siboehm.com/articles/22/CUDA-MMM
 *
 * width = column, height = row,
 * A[row][col], A[y][x]
 */

// Forward declaration of the matrix multiplication kernel
// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernelCoalescing(Matrix A, Matrix B, Matrix C)
{
    // Each thread reads one row of A and one column of B and computes the corresponding element of C
    float Cvalue = 0;
    int row = blockIdx.y * BLOCK_SIZE + (threadIdx.y % BLOCK_SIZE);
    int col = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    for (int k = 0; k < A.column; k++)
        Cvalue += A.elements[row * A.column + k] * B.elements[k * B.column + col];

    C.elements[row * C.column + col] = Cvalue;
}

namespace matmul
{
    // Matrix multiplication - Host code
    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void MatmulOperator::mat_mul_coalescing(const Matrix &A, const Matrix &B, Matrix &C)
    {

        for (int i = 0; i < C.row; i++)
            for (int j = 0; j < C.column; j++)
                C.elements[i * C.column + j] = 0;

        // Load A and B to device memory
        Matrix d_A;
        d_A.column = A.column;
        d_A.row = A.row;
        size_t size = A.column * A.row * sizeof(float);

        // Allocate memory
        cudaMalloc(&d_A.elements, size);

        // Copy data to GPU
        cudaMemcpy(d_A.elements, A.elements, size,
                   cudaMemcpyHostToDevice);
        Matrix d_B;
        d_B.column = B.column;
        d_B.row = B.row;
        size = B.column * B.row * sizeof(float);

        // Allocate memory
        cudaMalloc(&d_B.elements, size);
        // Copy data to GPU
        cudaMemcpy(d_B.elements, B.elements, size,
                   cudaMemcpyHostToDevice);

        // Allocate C in device memory
        Matrix d_C;
        d_C.column = C.column;
        d_C.row = C.row;
        size = C.column * C.row * sizeof(float);
        cudaMalloc(&d_C.elements, size);

        // Invoke kernel
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(C.column / dimBlock.x, C.row / dimBlock.y);
        std::cout << "Computing result using CUDA Kernel...\n";
        MatMulKernelCoalescing<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        // Read C from device memory
        cudaMemcpy(C.elements, d_C.elements, size,
                   cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }
}