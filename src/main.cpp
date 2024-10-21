#include "matmul.h"
#include <math.h>
#include <iostream>

#define BLK_SIZE 32
#define MAX_PRECISION_ERROR 1.e-6

#define A_ROW 640
#define A_COLUMN 1280
#define B_ROW 1280
#define B_COLUMN 640
#define C_ROW 640
#define C_COLUMN 640
#define NUM_THREAD 4

using namespace matmul;

void initialize_matrix(float A[], int size)
{
    for (int i = 0; i < size; i++)
    {
        A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

float interval_to_ms(struct timeval *start, struct timeval *end)
{
    float us_seconds = (end->tv_sec - start->tv_sec) * 1000000 + (end->tv_usec - start->tv_usec);
    return us_seconds / 1000;
}

bool check_identical(float native[], float output[], int size)
{
    for (int i = 0; i < size; i++)
    {
        if (abs((native[i] - output[i]) / (output[i])) > MAX_PRECISION_ERROR)
        {
            std::cout << "idex is" << i << native[i] << ", " << output[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{

    std::cout << "[Matrix Multiply] - Starting...\n";
    Matrix h_A;
    h_A.column = A_COLUMN;
    h_A.row = A_ROW;
    h_A.elements = new float[h_A.column * h_A.row];

    Matrix h_B;
    h_B.column = B_COLUMN;
    h_B.row = B_ROW;
    h_B.elements = new float[h_B.column * h_B.row];

    Matrix native_C;
    native_C.column = C_COLUMN;
    native_C.row = C_ROW;
    native_C.elements = new float[native_C.column * native_C.row];

    Matrix output_C;
    output_C.column = C_COLUMN;
    output_C.row = C_ROW;
    output_C.elements = new float[output_C.column * output_C.row];

    std::cout << "init start" << std::endl;
    initialize_matrix(h_A.elements, h_A.column * h_A.row);
    initialize_matrix(h_B.elements, h_B.column * h_B.row);
    std::cout << "init done" << std::endl;

    struct timeval start, end;
    int ms;

    MatmulOperator matmul_op = MatmulOperator();

    gettimeofday(&start, NULL);
    matmul_op.naive_mat_mul(h_A, h_B, native_C);
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    std::cout << "naive_mat_mul" << ": " << ms << " ms" << std::endl;

    gettimeofday(&start, NULL);
    matmul_op.mat_mul_reordering(h_A, h_B, output_C);
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    std::cout << "mat_mul_reordering" << ": " << ms << " ms" << std::endl;

    if (!check_identical(native_C.elements, output_C.elements, C_ROW * C_COLUMN))
    {
        std::cout << "incorrect output from mat_mul_unrolling\n"
                  << std::endl;
    }

    gettimeofday(&start, NULL);
    matmul_op.mat_mul_tiling(h_A, h_B, output_C, BLK_SIZE);
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    std::cout << "mat_mul_tiling" << ": " << ms << " ms" << std::endl;

    if (!check_identical(native_C.elements, output_C.elements, C_ROW * C_COLUMN))
    {
        std::cout << "incorrect output from mat_mul_tiling\n"
                  << std::endl;
    }

    gettimeofday(&start, NULL);
    matmul_op.mat_mul_cuda(h_A, h_B, output_C);
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    std::cout << "mat_mul_cuda" << ": " << ms << " ms" << std::endl;

    if (!check_identical(native_C.elements, output_C.elements, C_ROW * C_COLUMN))
    {
        std::cout << "incorrect output from mat_mul_cuda\n"
                  << std::endl;
    }

    gettimeofday(&start, NULL);
    matmul_op.mat_mul_cuda_shared(h_A, h_B, output_C);
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    std::cout << "mat_mul_cuda_shared" << ": " << ms << " ms" << std::endl;

    if (!check_identical(native_C.elements, output_C.elements, C_ROW * C_COLUMN))
    {
        std::cout << "incorrect output from mat_mul_cuda_shared\n"
                  << std::endl;
    }

    gettimeofday(&start, NULL);
    matmul_op.mat_mul_coalescing(h_A, h_B, output_C);
    gettimeofday(&end, NULL);
    ms = interval_to_ms(&start, &end);
    std::cout << "mat_mul_coalescing" << ": " << ms << " ms" << std::endl;

    if (!check_identical(native_C.elements, output_C.elements, C_ROW * C_COLUMN))
    {
        std::cout << "incorrect output from mat_mul_coalescing\n"
                  << std::endl;
    }
}