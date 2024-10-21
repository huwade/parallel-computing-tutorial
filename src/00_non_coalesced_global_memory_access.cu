#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.hpp"

/**
 * A[M][K] * B [K][N] = C[M][N]
 * A.elements[row * A.column + k] * B.elements[k * B.column + col];
 */

template <typename T>
__global__ void gemm_v00(size_t m, size_t n, size_t k, T alpha, T const *A,
                         size_t lda, T const *B, size_t ldb, T beta, T *C, size_t ldc)
{
    // compute the row and column of C

    // so non coalesced is row, col index change?
    size_t const C_row_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_col_idx{blockIdx.y * blockDim.y + threadIdx.y};

    if (C_row_idx < m && C_col_idx < n)
    {
        T sum{static_cast<T>(0)};
        for (size_t k_idx{0U}; k_idx < k; ++k_idx)
        {
            sum += A[C_row_idx * lda + k_idx] * B[k_idx * ldb + C_col_idx];
        }
        C[C_row_idx * ldc + C_col_idx] =
            alpha * sum + beta * C[C_row_idx * ldc + C_col_idx];
    }
}

template <typename T>
void launch_gemm_kernel_v00(size_t m, size_t n, size_t k, T const *alpha,
                            T const *A, size_t lda, T const *B, size_t ldb, T const *beta, T *C, size_t ldc, cudaStream_t stream)
{

    // The U suffix indicates that these are unsigned integers
    // (e.g., 32U is equivalent to static_cast<unsigned int>(32)
    dim3 const blockDim{32U, 32U, 1U};
    // dim3 const gridDim(CEIL_DIV(M, 32), CEIL_DEV(N, 32), 1);
    dim3 const grid_dim{
        (static_cast<unsigned int>(m) + block_dim.x - 1U) / block_dim.x,
        (static_cast<unsigned int>(n) + block_dim.y - 1U) / block_dim.y, 1U};

    // gemm_v00<T><<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
    gemm_v00<T><<<grid_dim, block_dim, 0U, stream>>>(m, n, k, *alpha, A, lda, B,
                                                     ldb, *beta, C, ldc);
}

template void launch_gemm_kernel_v00<float>(size_t m, size_t n, size_t k,
                                            float const *alpha,
                                            float const *A, size_t lda,
                                            float const *B, size_t ldb,
                                            float const *beta, float *C,
                                            size_t ldc, cudaStream_t stream);

template void launch_gemm_kernel_v00<double>(size_t m, size_t n, size_t k,
                                             double const *alpha,
                                             double const *A, size_t lda,
                                             double const *B, size_t ldb,
                                             double const *beta, double *C,
                                             size_t ldc, cudaStream_t stream);
template void launch_gemm_kernel_v00<__half>(size_t m, size_t n, size_t k,
                                             __half const *alpha,
                                             __half const *A, size_t lda,
                                             __half const *B, size_t ldb,
                                             __half const *beta, __half *C,
                                             size_t ldc, cudaStream_t stream);
