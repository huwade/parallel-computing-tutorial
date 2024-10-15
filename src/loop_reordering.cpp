#include "matmul.h"
#include <stdio.h>

namespace matmul
{
    void MatmulOperator::mat_mul_reordering(const Matrix &A, const Matrix &B, Matrix &C)
    {
        int i, j, k;
        float Aik;

        for (i = 0; i < C.row; i++)
        {
            for (k = 0; k < A.column; k++)
            {
                // Matrices are stored in row-major order:
                // M(row, col) = *(M.elements + row * M.width + col)
                float Aik = A.elements[i * A.column + k];
                for (j = 0; j < C.column; j++)
                {
                    C.elements[i * C.column + j] += Aik * B.elements[k * B.column + j];
                }
            }
        }
    }

}