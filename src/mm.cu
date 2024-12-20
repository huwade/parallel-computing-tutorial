#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#define BLOCK_DIM 32

#define checkCuda(val) check((val), #val, __FILE__, __LINE__)

template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
std::vector<T> create_rand_vector(size_t n)
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(-256, 256);

    std::vector<T> vec(n);

    for (size_t i{0}; i < n; i++)
    {
        vec.at(i) = static_cast<T>(uniform_dist(e));
    }

    return vec;
}

template <typename T>
bool allclose(std::vector<T> const &vec_1, std::vector<T> const &vec_2,
              T const &abs_tol)
{
    if (vec_1.size() != vec_2.size())
        return false;

    for (size_t i{0}; i < vec_1.size(); i++)
    {
        if (std::abs(vec_1.at(i) - vec_2.at(i)) > abs_tol)
        {
            std::cout << vec_1.at(i) << " " << vec_2.at(i) << std::endl;
            return false;
        }
    }
    return true;
}

// mat_1: m x n
// mat_2: n x p
// mat_3: m x p
// C[i][j] += A[i][k] * B[k][j]
template <typename T>
void mm(T const *mat_1, T const *mat_2, T *mat_3, size_t m, size_t n, size_t p)
{
    for (size_t i{0}; i < m; i++)
    {
        for (size_t j{0}; j < p; j++)
        {
            T acc_sum{static_cast<T>(0)};
            for (size_t k{0}; k < n; k++)
            {
                acc_sum += mat_1[i * n + k] * mat_2[k * p + j];
            }
            mat_3[i * p + j] = acc_sum;
        }
    }
}

template <typename T>
__global__ void mm_kernel(T const *mat_1, T const *mat_2, T *mat_3, size_t m,
                          size_t n, size_t p)
{
    // 2D block and 2D thread
    // Each thread computes one cell in mat_3.
    size_t C_row_idx{blockIdx.y * blockDim.y + threadIdx.y};
    size_t C_col_idx{blockIdx.x * blockDim.x + threadIdx.x};

    // Do not process outside the matrix.
    // Do not forget the equal sign!
    // mat_3: n x p
    if (C_row_idx < m && C_col_idx < p)
    {
        T acc_sum{static_cast<T>(0)};
        for (size_t k{0}; k < n; ++k)
        {
            acc_sum += mat_1[C_row_idx * n + k] * mat_2[k * p + C_col_idx];
        }
        mat_3[C_row_idx * p + C_col_idx] = acc_sum;
    }
}

template <typename T>
void mm_cuda(T const *mat_1, T const *mat_2, T *mat_3, size_t m, size_t n, size_t p)
{
    dim3 threads_per_block(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks_per_grid(1, 1);
    blocks_per_grid.x = std::ceil(static_cast<double>(p) /
                                  static_cast<double>(threads_per_block.x));

    blocks_per_grid.y = std::ceil(static_cast<double>(m) /
                                  static_cast<double>(threads_per_block.y));
    mm_kernel<<<blocks_per_grid, threads_per_block>>>(mat_1, mat_2, mat_3, m, n, p);
}

// mat_1: m x n
// mat_2: n x p
// mat_3: m x p
template <typename T>
bool random_test_mm_cuda(size_t m, size_t n, size_t p)
{
    std::vector<T> const mat_1_vec{create_rand_vector<T>(m * n)};
    std::vector<T> const mat_2_vec{create_rand_vector<T>(n * p)};
    // vector<int> vec(10); Creates a vector of size 10, all elements initialized to 0.
    // vector<int> vec{10}; Creates a vector with one element initialized to 10.
    std::vector<T> mat_3_vec(m * p);
    std::vector<T> mat_4_vec(m * p);
    T const *mat_1{mat_1_vec.data()};
    T const *mat_2{mat_2_vec.data()};
    T *mat_3{mat_3_vec.data()};
    T *mat_4{mat_4_vec.data()};

    mm(mat_1, mat_2, mat_3, m, n, p);

    T *d_mat_1, *d_mat_2, *d_mat_4;
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * mat_1_vec.size()));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * mat_2_vec.size()));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(T) * mat_4_vec.size()));

    checkCuda(cudaMemcpy(d_mat_1, mat_1, sizeof(T) * mat_1_vec.size(),
                         cudaMemcpyHostToDevice));

    checkCuda(cudaMemcpy(d_mat_2, mat_2, sizeof(T) * mat_2_vec.size(),
                         cudaMemcpyHostToDevice));

    mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
    cudaDeviceSynchronize();
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Copy data from device to host.
    checkCuda(cudaMemcpy(mat_4, d_mat_4, sizeof(T) * mat_4_vec.size(),
                         cudaMemcpyDeviceToHost));

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_4));

    return allclose<T>(mat_3_vec, mat_4_vec, 1e-4);
}

template <typename T>
bool random_multiple_test_mm_cuda(size_t num_tests)
{
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_int_distribution<int> uniform_dist(1, 256);

    size_t m{0}, n{0}, p{0};
    bool success{false};

    for (size_t i{0}; i < num_tests; i++)
    {
        m = static_cast<size_t>(uniform_dist(e));
        n = static_cast<size_t>(uniform_dist(e));
        p = static_cast<size_t>(uniform_dist(e));
        success = random_test_mm_cuda<T>(m, n, p);
        if (!success)
        {
            return false;
        }
    }
    return true;
}

template <typename T>
float measure_latency_mm_cuda(size_t m, size_t n, size_t p, size_t num_tests,
                              size_t num_warmups)
{
    cudaEvent_t startEvent, stopEvent;
    float time{0.0f};

    checkCuda(cudaEventCreate(&startEvent));
    checkCuda(cudaEventCreate(&stopEvent));

    T *d_mat_1, *d_mat_2, *d_mat_4;

    // Allocate device buffer.
    checkCuda(cudaMalloc(&d_mat_1, sizeof(T) * m * n));
    checkCuda(cudaMalloc(&d_mat_2, sizeof(T) * n * p));
    checkCuda(cudaMalloc(&d_mat_4, sizeof(T) * m * p));

    for (size_t i{0}; i < num_warmups; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
    }

    checkCuda(cudaEventRecord(startEvent, 0));
    for (size_t i{0}; i < num_tests; ++i)
    {
        mm_cuda(d_mat_1, d_mat_2, d_mat_4, m, n, p);
    }
    checkCuda(cudaEventRecord(stopEvent, 0));
    checkCuda(cudaEventSynchronize(stopEvent));
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Matrix Multiplication kernel failed to execute."
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
    checkCuda(cudaEventElapsedTime(&time, startEvent, stopEvent));

    // Free device buffer.
    checkCuda(cudaFree(d_mat_1));
    checkCuda(cudaFree(d_mat_2));
    checkCuda(cudaFree(d_mat_4));

    float latency{time / num_tests};

    return latency;
}

int main()
{
    constexpr size_t num_tests{10};

    assert(random_multiple_test_mm_cuda<int32_t>(num_tests));
    assert(random_multiple_test_mm_cuda<float>(num_tests));
    assert(random_multiple_test_mm_cuda<double>(num_tests));
    constexpr size_t num_measurement_tests{10};
    constexpr size_t num_measurement_warmups{10};
    size_t m{1024}, n{1024}, p{1024};

    float mm_cuda_int32_latency{measure_latency_mm_cuda<int32_t>(
        m, n, p, num_measurement_tests, num_measurement_warmups)};
    float mm_cuda_float_latency{measure_latency_mm_cuda<float>(
        m, n, p, num_measurement_tests, num_measurement_warmups)};
    float mm_cuda_double_latency{measure_latency_mm_cuda<double>(
        m, n, p, num_measurement_tests, num_measurement_warmups)};

    std::cout << "Matrix Multiplication CUDA Latency" << std::endl;
    std::cout << "m: " << m << " "
              << "n: " << n << " "
              << "p: " << p << std::endl;
    std::cout << "INT32: " << std::fixed << std::setprecision(5)
              << mm_cuda_int32_latency << " ms" << std::endl;
    std::cout << "FLOAT: " << std::fixed << std::setprecision(5)
              << mm_cuda_float_latency << " ms" << std::endl;
    std::cout << "DOUBLE: " << std::fixed << std::setprecision(5)
              << mm_cuda_double_latency << " ms" << std::endl;
}
