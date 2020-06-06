#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <stdlib.h>
#include "filter.h"
#include <intrin.h>

enum {
    NONE        = 0x0000,
    SSE2        = 0x0001,
    SSE3        = 0x0002,
    SSSE3       = 0x0004,
    SSE41       = 0x0008,
    SSE42       = 0x0010,
    POPCNT      = 0x0020,
    XOP         = 0x0040,
    AVX         = 0x0080,
    AVX2        = 0x0100,
    FMA3        = 0x0200,
    FMA4        = 0x0400,
    FAST_GATHER = 0x00800,
    AVX2FAST    = 0x001000,
    AVX512F     = 0x002000,
    AVX512DQ    = 0x004000,
    AVX512IFMA  = 0x008000,
    AVX512PF    = 0x010000,
    AVX512ER    = 0x020000,
    AVX512CD    = 0x040000,
    AVX512BW    = 0x080000,
    AVX512VL    = 0x100000,
    AVX512VBMI  = 0x200000,
    AVX512VNNI  = 0x400000,
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
    if (simd & AVX) {
        if (CPUInfo[1] & 0x00000020)
            simd |= AVX2;
        if (CPUInfo[1] & (1<<18)) //rdseed -> Broadwell
            simd |= FAST_GATHER;
        if ((simd & AVX) && ((xgetbv >> 5) & 7) == 7) {
            if (CPUInfo[1] & (1u << 16)) simd |= AVX512F;
            if (simd & AVX512F) {
                if (CPUInfo[1] & (1u << 17)) simd |= AVX512DQ;
                if (CPUInfo[1] & (1u << 21)) simd |= AVX512IFMA;
                if (CPUInfo[1] & (1u << 26)) simd |= AVX512PF;
                if (CPUInfo[1] & (1u << 27)) simd |= AVX512ER;
                if (CPUInfo[1] & (1u << 28)) simd |= AVX512CD;
                if (CPUInfo[1] & (1u << 30)) simd |= AVX512BW;
                if (CPUInfo[1] & (1u << 31)) simd |= AVX512VL;
                if (CPUInfo[2] & (1u <<  1)) simd |= AVX512VBMI;
                if (CPUInfo[2] & (1u << 11)) simd |= AVX512VNNI;
            }
        }
        __cpuid(CPUInfo, 0x80000001);
        if (CPUInfo[2] & 0x00000800)
            simd |= XOP;
        if (CPUInfo[2] & 0x00010000)
            simd |= FMA4;
    }
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
void multi_thread_func_avx512(int thread_id, int thread_num, void *param1, void *param2);

typedef struct {
    MULTI_THREAD_FUNC func[3];
    DWORD simd;
} EDGELEVEL_FUNC;

static const EDGELEVEL_FUNC FUNC_LIST[] = {
    { { multi_thread_check_threshold, multi_thread_func_avx512, multi_thread_func_avx512        }, AVX512BW|AVX512F|AVX2|AVX },
    { { multi_thread_check_threshold, multi_thread_func_avx2,   multi_thread_func_avx2          }, AVX2|AVX },
    { { multi_thread_check_threshold, multi_thread_func_avx,    multi_thread_func_avx_aligned   }, AVX },
    { { multi_thread_check_threshold, multi_thread_func_sse41,  multi_thread_func_sse41_aligned }, SSE41|SSSE3|SSE2 },
    { { multi_thread_check_threshold, multi_thread_func_ssse3,  multi_thread_func_ssse3_aligned }, SSSE3|SSE2 },
    { { multi_thread_check_threshold, multi_thread_func_sse2,   multi_thread_func_sse2_aligned  }, SSE2 },
    { { multi_thread_check_threshold, multi_thread_func,        multi_thread_func               }, NONE },
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
