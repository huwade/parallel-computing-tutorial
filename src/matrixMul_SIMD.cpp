#include "matmul.h"
#include <stdio.h>
#include <assert.h>
#ifdef __SSE__
#include <xmmintrin.h> // intel SSE intrinsic
#endif
#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define MAX_TRANSPOSE_BUFFER 10 * 1024 * 1024

namespace matmul
{
    inline void simd_mul_fp_128(const float &a, const float &b, float &c)
    {
#ifdef __SSE__
        __m128 val = _mm_mul_ps(_mm_load_ps(a), _mm_load_ps(b));
#endif
    }
}