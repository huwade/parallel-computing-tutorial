#include "matmul.h"
#include <stdio.h>

namespace matmul
{

    void MatmulOperator::mat_mul_reordering(const Matrix &A, const Matrix &B, Matrix &C)
    {
        int i, j, k;
        float Aik;
        // C[i][j] += A[i][k] * B[k][j]
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
/*
Example 1:

Input: nums = [1,-2,3,-2]
Output: 3
Explanation: Subarray [3] has maximum sum 3.
Example 2:

Input: nums = [5,-3,5]
Output: 10
Explanation: Subarray [5,5] has maximum sum 5 + 5 = 10.
Example 3:

Input: nums = [-3,-2,-3]
Output: -2
Explanation: Subarray [-2] has maximum sum -2.
*/
