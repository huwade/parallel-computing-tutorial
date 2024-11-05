#include <cuda_runtime.h>
#include "matmul.h"
#include <iostream>
#define BLOCK_SIZE 32

// Matrix multiplication kernel called by MatMul()
__global__ void matrixMultiplyShared_coales(Matrix A, Matrix B, Matrix C)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float Cvalue = 0;
    // Thread row and column within Csub
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    for (int m = 0; m < (A.column / BLOCK_SIZE); ++m)
    {
        As[tx][ty] = A.elements[row * A.column + m * BLOCK_SIZE + tx];
        Bs[tx][ty] = B.elements[(m * BLOCK_SIZE + ty) * A.column + col];

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[tx][e] * Bs[e][ty];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    C.elements[row * A.column + col] = Cvalue;
}

namespace matmul
{
    void MatmulOperator::mat_mul_coalescing(const Matrix &A, const Matrix &B, Matrix &C)
    {
        // Load A and B to device memory
        Matrix d_A;
        d_A.column = A.column;
        d_A.row = A.row;
        size_t size = A.column * A.row * sizeof(float);
        cudaMalloc(&d_A.elements, size);
        cudaMemcpy(d_A.elements, A.elements, size,
                   cudaMemcpyHostToDevice);
        Matrix d_B;
        d_B.column = B.column;
        d_B.row = B.row;
        size = B.column * B.row * sizeof(float);
        cudaMalloc(&d_B.elements, size);
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
        matrixMultiplyShared_coales<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
        // Read C from device memory
        cudaMemcpy(C.elements, d_C.elements, size,
                   cudaMemcpyDeviceToHost);
        // Free device memory
        cudaFree(d_A.elements);
        cudaFree(d_B.elements);
        cudaFree(d_C.elements);
    }
}
