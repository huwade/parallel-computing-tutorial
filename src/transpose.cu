/**
 * To understand CUDA Coalesced Memory Access
 * Reference: https://leimao.github.io/blog/CUDA-Coalesced-Memory-Access/
 *
 * CUDA Matrix Transpose
 *
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#define CHECH_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, char const *func, char const *file, int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() check_last(__FILE__, __LINE__)
void check_last(char const *file, int line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, size_t num_repeats = 10, size_t num_warmups = 10)
{

    cudaEvent_t start, stop;
    float time;

    CHECH_CUDA_ERROR(cudaEventCreate(&start));
    CHECH_CUDA_ERROR(cudaEventCreate(&stop));

    for (size_t i{0}; i < num_warmups; i++)
    {
        bound_function(stream);
    }

    CHECH_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECH_CUDA_ERROR(cudaEventRecord(start, stream));
    for (size_t i{0}; i < num_repeats; i++)
    {
        bound_function(stream);
    }
    CHECH_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECH_CUDA_ERROR(cudaEventSynchronize(stop);)
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    float const latency{time / num_repeats};

    return latency;
}

constexpr size_t div_up(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

template <typename T>
__global__ void transpose_read_coalesced(T *output_matrix, T const *input_matrix, size_t M, size_t N)
{
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};

    size_t const from_idx{i * N + j};
    if ((i < M) && (j < N))
    {
        size_t const to_idx{j * M + i};
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

template <typename T>
__global__ void transpose_write_coalesced(T *output_matrix, T const *input_matrix, size_t M, size_t N)
{
    size_t const j{threadIdx.x + blockIdx.x * blockDim.x};
    size_t const i{threadIdx.y + blockIdx.y * blockDim.y};

    size_t const to_idx{i * N + j};
    if ((i < M) && (j < N))
    {
        size_t const from_idx{j * M + i};
        output_matrix[to_idx] = input_matrix[from_idx];
    }
}

template <typename T>
void launch_transpose_read_coalesced(T *output_matrix, T const *input_matrix,
                                     size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(N, warp_size)),
                               static_cast<unsigned int>(div_up(M, warp_size))};
    transpose_read_coalesced<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}

template <typename T>
void launch_transpose_write_coalesced(T *output_matrix, T const *input_matrix,
                                      size_t M, size_t N, cudaStream_t stream)
{
    constexpr size_t const warp_size{32};
    dim3 const threads_per_block{warp_size, warp_size};
    dim3 const blocks_per_grid{static_cast<unsigned int>(div_up(M, warp_size)),
                               static_cast<unsigned int>(div_up(N, warp_size))};
    transpose_write_coalesced<<<blocks_per_grid, threads_per_block, 0,
                                stream>>>(output_matrix, input_matrix, M, N);
    CHECK_LAST_CUDA_ERROR();
}