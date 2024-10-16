#include <sys/time.h>

typedef struct
{
    int row;
    int column;
    float *elements;
} Matrix;

struct thread_args
{
    const Matrix *A;
    const Matrix *B;
    const Matrix *C;
    int start_i, end_i, blk_size;
};

typedef struct
{
    int blk_size;
    int num_thread = 8;
} optimization_params;

typedef struct
{
    Matrix A, B, C;
    optimization_params opt_params;
} matmul_params;

namespace matmul
{

    class MatmulOperator
    {
    public:
        void naive_mat_mul(const Matrix &A, const Matrix &B, Matrix &C);
        void mat_mul_reordering(const Matrix &A, const Matrix &B, Matrix &C);
        void mat_mul_tiling(const Matrix &A, const Matrix &B, Matrix &C, int BLK_SIZE);
        void mat_mul_cuda(const Matrix &A, const Matrix &B, Matrix &C);
        void mat_mul_cuda_shared(const Matrix &A, const Matrix &B, Matrix &C);
    };
}
