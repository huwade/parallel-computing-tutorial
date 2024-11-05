#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

template <typename T>
__global__ void gemm_v01(size_t m, size_t n, size_t k, T alpha, T const *A,
                         size_t lda, T const *B, size_t ldb, T beta, T *C,
                         size_t ldc)
{
    // compute the row and column of C that this thread is responsible for.
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; k_idx++)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }

        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}