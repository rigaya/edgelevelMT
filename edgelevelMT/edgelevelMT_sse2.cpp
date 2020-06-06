#define USE_SSE2  1
#define USE_SSSE3 0
#define USE_SSE41 0
#define USE_AVX   0
#define USE_AVX2  0

#include "edgelevelMT_simd.h"

void multi_thread_func_sse2_aligned(int thread_id, int thread_num, void *param1, void *param2) {
	multi_thread_func_simd<true>(thread_id, thread_num, param1, param2);
}

void multi_thread_func_sse2(int thread_id, int thread_num, void *param1, void *param2) {
	multi_thread_func_simd<false>(thread_id, thread_num, param1, param2);
}
