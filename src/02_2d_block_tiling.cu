#include <cuda_fp16.h>

#include "cuda_gemm.hpp"
#include "cuda_gemm_utils.cuh"
#include "cuda_gemm_utils.hpp"

template <typename T, size_t BOLCK_TILE_SIZE_X, size_t, BOLCK_TILE_SIZE_Y, size_t BOLCK_TILE_SIZE_Z>
__global__ void gemm_v02(size_t m, size_t n, size_t k, T alpha, T const *A,
                         size_t lda, T const *B, size_t ldb, T beta, T *C,
                         size_t ldc)
{

    /**
     * Avoid using blockDim.x * blockDim.y as the number of threads per block.
     * Because it is a runtime constant and the compiler can't optimize the loop unrolling based on it.
     * Use a compile time constant instead.
     */

    constexpr size_t NUM_THREADS{BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y};
    size_t const thread_linear_idx{threadIdx.y * blockDim.x + threadIdx.x};
}