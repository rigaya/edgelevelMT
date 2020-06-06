#pragma once

#include <Windows.h>
#include "filter.h"
#include <intrin.h>

enum {
    NONE   = 0x0000,
    SSE2   = 0x0001,
    SSE3   = 0x0002,
    SSSE3  = 0x0004,
    SSE41  = 0x0008,
    SSE42  = 0x0010,
    POPCNT = 0x0020,
    AVX    = 0x0040,
    AVX2   = 0x0080,
    FMA3   = 0x0100,
};

static DWORD get_availableSIMD() {
    int CPUInfo[4];
    __cpuid(CPUInfo, 1);
    DWORD simd = NONE;
    if (CPUInfo[3] & 0x04000000) simd |= SSE2;
    if (CPUInfo[2] & 0x00000001) simd |= SSE3;
    if (CPUInfo[2] & 0x00000200) simd |= SSSE3;
    if (CPUInfo[2] & 0x00080000) simd |= SSE41;
    if (CPUInfo[2] & 0x00100000) simd |= SSE42;
    if (CPUInfo[2] & 0x00800000) simd |= POPCNT;
#if (_MSC_VER >= 1600)
    UINT64 xgetbv = 0;
    if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
        xgetbv = _xgetbv(0);
        if ((xgetbv & 0x06) == 0x06)
            simd |= AVX;
#if (_MSC_VER >= 1700)
        if(CPUInfo[2] & 0x00001000 )
            simd |= FMA3;
#endif //(_MSC_VER >= 1700)
    }
#endif
#if (_MSC_VER >= 1700)
    __cpuid(CPUInfo, 7);
    if ((simd & AVX) && (CPUInfo[1] & 0x00000020))
        simd |= AVX2;
#endif
    return simd;
}


//---------------------------------------------------------------------
//      エッジレベル調整 関数リスト
//---------------------------------------------------------------------
void multi_thread_check_threshold(int thread_id, int thread_num, void *param1, void *param2);

void multi_thread_func( int thread_id, int thread_num, void *param1, void *param2 );
void multi_thread_func_sse2(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_sse2_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_ssse3_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_sse41(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_sse41_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_avx(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_avx_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_avx2(int thread_id, int thread_num, void *param1, void *param2);

typedef struct {
    MULTI_THREAD_FUNC func[3];
    DWORD simd;
} EDGELEVEL_FUNC;

static const EDGELEVEL_FUNC FUNC_LIST[] = {
    { { multi_thread_check_threshold, multi_thread_func_avx2,  multi_thread_func_avx2          }, AVX2|AVX },
    { { multi_thread_check_threshold, multi_thread_func_avx,   multi_thread_func_avx_aligned   }, AVX },
    { { multi_thread_check_threshold, multi_thread_func_sse41, multi_thread_func_sse41_aligned }, SSE41|SSSE3|SSE2 },
    { { multi_thread_check_threshold, multi_thread_func_ssse3, multi_thread_func_ssse3_aligned }, SSSE3|SSE2 },
    { { multi_thread_check_threshold, multi_thread_func_sse2,  multi_thread_func_sse2_aligned  }, SSE2 },
    { { multi_thread_check_threshold, multi_thread_func,       multi_thread_func               }, NONE },
};

const MULTI_THREAD_FUNC *get_func_list() {
    DWORD simd_avail = get_availableSIMD();
    for (int i = 0; i < _countof(FUNC_LIST); i++) {
        if ((FUNC_LIST[i].simd & simd_avail) == FUNC_LIST[i].simd) {
            return FUNC_LIST[i].func;
        }
    }
    return NULL;
}
