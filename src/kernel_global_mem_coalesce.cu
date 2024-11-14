// CUDA runtime
#include <cuda_runtime.h>
#include "matmul.h"
#include <iostream>

/**
 * This is from https://siboehm.com/articles/22/CUDA-MMM
 */

#define BLOCKSIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

__global__ void sgemm_global_mem_coalesce(Matrix A, Matrix B, Matrix C)
{
    // Each thread reads one row of A and one column of B and computes the corresponding element of C

    size_t const C_row_idx{blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE)};
    size_t const C_col_idx{blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE)};

    // each thread computes one element of C if in range
    if (C_row_idx < A.row && C_col_idx < A.column)
    {
        float Cvalue = 0;
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
        dim3 dimBlock(32 * 32);
        dim3 dimGrid(CEIL_DIV(C.column, 32), CEIL_DIV(C.row, 32));
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


* Column Index (x1 = threadIdx.x % 8)
* Row    Index (y1 = threadIdx.x / 8)

threadIdx.x = 0  → (x1, y1) = (0 % 8, 0 / 8) → (0, 0)
threadIdx.x = 1  → (x1, y1) = (1 % 8, 1 / 8) → (1, 0)
threadIdx.x = 2  → (x1, y1) = (2 % 8, 2 / 8) → (2, 0)
...
threadIdx.x = 7  → (x1, y1) = (7 % 8, 7 / 8) → (7, 0)
threadIdx.x = 8  → (x1, y1) = (8 % 8, 8 / 8) → (0, 1)
threadIdx.x = 9  → (x1, y1) = (9 % 8, 9 / 8) → (1, 1)
...
threadIdx.x = 15 → (x1, y1) = (15 % 8, 15 / 8) → (7, 1)
threadIdx.x = 16 → (x1, y1) = (16 % 8, 16 / 8) → (0, 2)
...
threadIdx.x = 31 → (x1, y1) = (31 % 8, 31 / 8) → (7, 3)
 */