// CUDA runtime
#include <cuda_runtime.h>
#include "matmul.h"
#include <iostream>
#define BLOCK_SIZE 32

/**
 * This is from https://siboehm.com/articles/22/CUDA-MMM
 */

#define BLOCKSIZE 32

__global__ void sgemm_global_mem_coalesce(Matrix A, Matrix B, Matrix C)
{
    // Each thread reads one row of A and one column of B and computes the corresponding element of C
    float Cvalue = 0;
    size_t const C_row_idx{blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)};
    size_t const C_col_idx{blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)};

    // each thread computes one element of C if in range
    if (C_row_idx < A.row && C_col_idx < A.column)
    {
        for (int k = 0; k < A.column; k++)
            Cvalue += A.elements[C_row_idx * A.column + k] * B.elements[k * B.column + C_col_idx];

        C.elements[C_row_idx * C.column + C_col_idx] = Cvalue;
    }
}

namespace matmul
{
    // Matrix multiplication - Host code
    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void MatmulOperator::mat_mul_global_mem_coalesce(const Matrix &A, const Matrix &B, Matrix &C)
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
        sgemm_global_mem_coalesce<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        // Read C from device memory
        cudaMemcpy(C.elements, d_C.elements, size,
                   cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }
}

/**
 * template <const uint BLOCKSIZE>
 * __global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                            const float *A, const float *B,
                                            float beta, float *C)
    {
        const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
        const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

        // if statement is necessary to make things work under tile quantization
        if (cRow < M && cCol < N)
        {
            float tmp = 0.0;
            for (int i = 0; i < K; ++i)
            {
                tmp += A[cRow * K + i] * B[i * N + cCol];
            }
            C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
        }
    }


 * threadIdx.x (max 3) / blocksize (is 4)
 * floor (0 / 4) = 0
 * 1 / 4 = 0
 * 2 / 4 = 0
 * 3 / 4 = 0
 * 4 / 4 = 1
 * 5 / 4 = 1
 * 6 / 4 = 1
 * 7 / 4 = 1
 * 8 / 4 = 2
 *
 * threadIdx.x % blocksize
 * 0 % 4 = 0
 * 1 % 4 = 1
 * 2 % 4 = 2
 * 3 % 4 = 3
 * 4 % 4 = 0
 * 5 % 4 = 1
 * 6 % 4 = 2
 * 7 % 4 = 3
 * 8 % 4 = 0
 */