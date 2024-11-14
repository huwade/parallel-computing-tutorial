#ifndef CUDA_GEMM_UTILS_CUH
#define CUDA_GEMM_UTILS_CUH

#include <cuda_runtime.h>
#include "cuda_gemm_utils.hpp"

template <typename T, size_t BLOCK_SIZE_X, size_t BLOCK_SIZE_Y, size_t BLOCK_SIZE_K,
          size_t NUM_THREADS, size_t BLOCK_TILE_SKEW_SIZE_X = 0U, size_t BLOCK_TILE_SKEW_SIZE_K = 0U>
__device__ void load_data_from_global_memory_to_shared_memory()