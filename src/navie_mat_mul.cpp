#include "matmul.h"

#include <sys/time.h>
#include <string>
#include <stdio.h>
#include <math.h>
#include <assert.h>

namespace matmul
{
    void MatmulOperator::naive_mat_mul(const Matrix &A, const Matrix &B, Matrix &C)
    {
        int i, j, k;

        // Naive matrix multiplication
        for (int i = 0; i < C.row; i++)
        {
            for (int j = 0; j < C.column; j++)
            {
                float acc = 0.0f;
                for (int k = 0; k < A.column; k++)
                {
                    // Corrected indexing for 1D array
                    acc += A.elements[i * A.column + k] * B.elements[k * B.column + j];
                }
                C.elements[i * C.column + j] = acc; // Assign the result to C
            }
        }
    }
}
