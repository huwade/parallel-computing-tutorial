#include "matmul.h"
#include <stdio.h>
#include <assert.h>

namespace matmul
{
    void MatmulOperator::mat_mul_tiling(const Matrix &A, const Matrix &B, Matrix &C, int block_size)
    {

        for (int i = 0; i < C.row; i++)
            for (int j = 0; j < C.column; j++)
                C.elements[i * C.column + j] = 0;

        for (int ti = 0; ti < C.row; ti += block_size)
        {
            for (int tk = 0; tk < A.column; tk += block_size)
            {
                for (int tj = 0; tj < C.column; tj += block_size)
                {
                    for (int i = ti; i < ti + block_size; i++)
                    {
                        for (int k = tk; k < tk + block_size; k++)
                        {
                            float Aik = A.elements[i * A.column + k];
                            for (int j = tj; j < tj + block_size; j++)
                            {
                                C.elements[i * C.column + j] += Aik * B.elements[k * B.column + j];
                            }
                        }
                    }
                }
            }
        }
    }
}