# This is a reimplementation of the MIT course 'Parallel Computing Tutorial' and an optimization of matrix multiplication

## Reference : <https://github.com/mit-han-lab/parallel-computing-tutorial>

## Reference : <https://siboehm.com/articles/22/CUDA-MMM>

## Reference : <https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/>

## Reference : <https://www.youtube.com/watch?v=86FAWCzIe_4>

| Method                  | Time (ms) |
|-------------------------|-----------|
| naive_mat_mul           | 3617      |
| mat_mul_reordering      | 2220      |
| mat_mul_tiling          | 2277      |
| mat_mul_cuda            | 669       |
| mat_mul_cuda_shared     | 15        |
| mat_mul_cuda_coalescing | 0         |

![alt text](image.png)

一個thread負責一個 output matrix 的值的計算
'''cpp
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread reads one row of A and one column of B and computes the corresponding element of C
    float Cvalue = 0;
    int C_row_idx = blockIdx.y *blockDim.y + threadIdx.y;
    int C_col_idx = blockIdx.x* blockDim.x + threadIdx.x;

    // each thread computes one element of C if in range
    if (C_row_idx < A.row && C_col_idx < A.column)
    {
        for (int k = 0; k < C.column; k++)
            Cvalue += A.elements[C_row_idx * A.column + k] * B.elements[k * B.column + C_col_idx];

        C.elements[C_row_idx * C.column + C_col_idx] = Cvalue;
    }
}
'''
