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

    size_t const C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};
    size_t const C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};

    __shared__ T A_thread_block_tile[BLOCK_TILE_SIZE_Y][BLOCK_TILE_SIZE_K];
    __shared__ T B_thread_block_tile[BLOCK_TILE_SIZE_K][BLOCK_TILE_SIZE_X];

    size_t const num_thread_block_tiles{(k + BOLCK_TILE_SIZE_K - 1) / BOLCK_TILE_SIZE_K};

    sum{static_cast<T>(0)};

    for (size_t thread_block_tile_idx{0U};
         thread_block_tile_idx < num_thread_block_tiles;
         thread_block_tile_idx++)
    {
        load_data_from_global_memory_to_shared_memory<
            T, BLOCK_TILE_SIZE_X, BLOCK_TILE_SIZE_Y, BLOCK_TILE_SIZE_K,
            NUM_THREADS>(A, lda, B, ldb, A_thread_block_tile,
                         B_thread_block_tile, thread_block_tile_idx,
                         thread_linear_idx, m, n, k);
        __syncthreads();
    }
}