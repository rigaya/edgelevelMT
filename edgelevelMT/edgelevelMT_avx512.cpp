#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "filter.h"
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

#define PIXELYC_SIZE 6

#define SUFFLE_YCP_Y       _mm512_load_si512((__m512i*)Array_SUFFLE_YCP_Y)
#define MASK_YCP_SELECT_Y  _mm512_load_si512((__m512i*)Array_MASK_YCP_SELECT_Y)
#define MASK_FRAME_EDGE    _mm_load_si128((__m128i*)Array_MASK_FRAME_EDGE)

//_mm512_srli_si512, _mm512_slli_si512は
//単に128bitシフト×2をするだけの命令である
#define _mm512_bsrli_epi128 _mm512_srli_si512
#define _mm512_bslli_epi128 _mm512_slli_si512

template<bool aligned_store>
static void __forceinline memcpy_avx512(void *_dst, void *_src, int size) {
    uint8_t *dst = (uint8_t *)_dst;
    uint8_t *src = (uint8_t *)_src;
    if (size < 256) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    uint8_t *dst_fin = dst + size;
    uint8_t *dst_aligned_fin = (uint8_t *)(((size_t)(dst_fin + 63) & ~63) - 256);
    __m512i z0, z1, z2, z3;
    const int start_align_diff = (int)((size_t)dst & 63);
    if (start_align_diff) {
        z0 = _mm512_loadu_si512((__m512i*)src);
        _mm512_storeu_si512((__m512i*)dst, z0);
        dst += 64 - start_align_diff;
        src += 64 - start_align_diff;
    }
    for (; dst < dst_aligned_fin; dst += 256, src += 256) {
        z0 = _mm512_loadu_si512((__m512i*)(src +   0));
        z1 = _mm512_loadu_si512((__m512i*)(src +  64));
        z2 = _mm512_loadu_si512((__m512i*)(src + 128));
        z3 = _mm512_loadu_si512((__m512i*)(src + 192));
        _mm512_storeu_si512((__m512i*)(dst +   0), z0);
        _mm512_storeu_si512((__m512i*)(dst +  64), z1);
        _mm512_storeu_si512((__m512i*)(dst + 128), z2);
        _mm512_storeu_si512((__m512i*)(dst + 192), z3);
    }
    uint8_t *dst_tmp = dst_fin - 256;
    src -= (dst - dst_tmp);
    z0 = _mm512_loadu_si512((__m512i*)(src +   0));
    z1 = _mm512_loadu_si512((__m512i*)(src +  64));
    z2 = _mm512_loadu_si512((__m512i*)(src + 128));
    z3 = _mm512_loadu_si512((__m512i*)(src + 192));
    _mm512_storeu_si512((__m512i*)(dst_tmp +   0), z0);
    _mm512_storeu_si512((__m512i*)(dst_tmp +  64), z1);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 128), z2);
    _mm512_storeu_si512((__m512i*)(dst_tmp + 192), z3);
}

template<bool avx512vbmi>
__m512i __forceinline load_y_from_yc48(const BYTE *src) {
    alignas(64) static const uint16_t PACK_YC48_SHUFFLE_AVX512[32] = {
         0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
        48, 51, 54, 57, 60, 63,  2,  5,  8, 11, 14, 17, 20, 23, 26, 29
    };
    alignas(64) static const uint8_t PACK_YC48_SHUFFLE_AVX512_VBMI[64] = {
         0,   1,   6,   7,  12,  13,  18,  19,  24,  25,  30,  31, 36, 37, 42, 43, 48, 49, 54, 55, 60, 61, 66, 67, 72, 73, 78, 79, 84, 85, 90, 91,
        96,  97, 102, 103, 108, 109, 114, 115, 120, 121, 126, 127,  4,  5, 10, 11, 16, 17, 22, 23, 28, 29, 34, 35, 40, 41, 46, 47, 52, 53, 58, 59
    };
    __m512i z0 = _mm512_load_si512(avx512vbmi ? (__m512i *)PACK_YC48_SHUFFLE_AVX512_VBMI : (__m512i *)PACK_YC48_SHUFFLE_AVX512);
    __m512i z5 = _mm512_loadu_si512((__m512i *)(src +   0));
    __m512i z4 = _mm512_loadu_si512((__m512i *)(src +  64));
    __m512i z3 = _mm512_loadu_si512((__m512i *)(src + 128));
    __m512i z1 = z0;
    __mmask32 k7 = 0xffc00000;
    if (avx512vbmi) {
        z1 = _mm512_permutex2var_epi8(z5/*a*/, z1/*idx*/, z4/*b*/);
        z1 = _mm512_mask_mov_epi16(z1, k7, _mm512_permutexvar_epi8(z0/*idx*/, z3));
    } else {
        z1 = _mm512_permutex2var_epi16(z5/*a*/, z1/*idx*/, z4/*b*/);
#if 1 //どちらでもあまり速度は変わらない
        z1 = _mm512_mask_permutexvar_epi16(z1/*src*/, k7, z0/*idx*/, z3);
#else
        z1 = _mm512_mask_mov_epi16(z1, k7, _mm512_permutexvar_epi16(z0/*idx*/, z3));
#endif
    }
    return z1;
}

static void __forceinline insert_y_yc48(uint8_t *dst, const uint8_t *src, const __m512i& zY) {
    alignas(64) static const uint16_t shuffle_yc48[] = {
         0, 11, 22,  1, 12, 23,  2, 13, 24,  3, 14, 25,  4, 15, 26,  5,
        16, 27,  6, 17, 28,  7, 18, 29,  8, 19, 30,  9, 20, 31, 10, 21

    };
    __m512i z0 = _mm512_loadu_si512((__m512i *)(src +   0));
    __m512i z1 = _mm512_loadu_si512((__m512i *)(src +  64));
    __m512i z2 = _mm512_loadu_si512((__m512i *)(src + 128));

    __m512i z7 = _mm512_permutexvar_epi16(_mm512_loadu_si512((__m512i *)(shuffle_yc48)), zY);

    __mmask32 k1 = 0x92492492u;
    __mmask32 k0 = k1 >> 1;
    __mmask32 k2 = k1 >> 2;
    z0 = _mm512_mask_mov_epi16(z0, k0, z7);
    z1 = _mm512_mask_mov_epi16(z1, k1, z7);
    z2 = _mm512_mask_mov_epi16(z2, k2, z7);

    _mm512_storeu_si512((dst +   0), z0);
    _mm512_storeu_si512((dst +  64), z1);
    _mm512_storeu_si512((dst + 128), z2);
}

#pragma warning (push)
#pragma warning (disable:4127) //warning  C4127: 条件式が定数です。
template<int shift>
static __forceinline __m512i _mm512_alignr512_epi8(const __m512i &z1, const __m512i &z0) {
    static_assert(0 <= shift && shift <= 64, "0 <= shift && shift <= 64");
    if (shift == 0) {
        return z0;
    } else if (shift == 64) {
        return z1;
    } else if (shift % 4 == 0) {
        return _mm512_alignr_epi32(z1, z0, shift / 4);
    } else if (shift <= 16) {
        __m512i z01 = _mm512_alignr_epi32(z1, z0, 4);
        return _mm512_alignr_epi8(z01, z0, shift);
    } else if (shift <= 32) {
        __m512i z010 = _mm512_alignr_epi32(z1, z0, 4);
        __m512i z011 = _mm512_alignr_epi32(z1, z0, 8);
        return _mm512_alignr_epi8(z011, z010, std::max(shift - 16, 0));
    } else if (shift <= 48) {
        __m512i z010 = _mm512_alignr_epi32(z1, z0, 8);
        __m512i z011 = _mm512_alignr_epi32(z1, z0, 12);
        return _mm512_alignr_epi8(z011, z010, std::max(shift - 32, 0));
    } else { //shift <= 64
        __m512i z01 = _mm512_alignr_epi32(z1, z0, 12);
        return _mm512_alignr_epi8(z1, z01, std::max(shift - 48, 0));
    }
}
#pragma warning (pop)

static __forceinline __m512i get_previous_2_y_pixels(BYTE *src) {
    static const _declspec(align(16)) BYTE SHUFFLE_LAST_2_Y_PIXELS[] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x04, 0x05, 0x0a, 0x0b
    };
    __m128i x0 = _mm_loadu_si128((__m128i *)(src - 16));
    x0 = _mm_shuffle_epi8(x0, _mm_load_si128((__m128i *)SHUFFLE_LAST_2_Y_PIXELS));
    __m512i z0 = _mm512_setzero_si512();
    return _mm512_inserti32x4(z0, x0, 3);
}

static __forceinline __m512i edgelevel_avx512_32(
    const __m512i& zSrc1YM2, const __m512i &zSrc1YM1,
    const __m512i& zSrc0, const __m512i&zSrc1, const __m512i &zSrc2,
    const __m512i &zSrc1YP1, const __m512i &zSrc1YP2,
    const __m512i &zThreshold, const __m512i &zStrength, const __m512i &zWc, const __m512i &zBc) {
    __m512i zVmax = _mm512_max_epi16(zSrc1, zSrc1YM2);
    __m512i zVmin = _mm512_min_epi16(zSrc1, zSrc1YM2);
    zVmax = _mm512_max_epi16(zVmax, zSrc1YM1);
    zVmin = _mm512_min_epi16(zVmin, zSrc1YM1);
    zVmax = _mm512_max_epi16(zVmax, zSrc1YP1);
    zVmin = _mm512_min_epi16(zVmin, zSrc1YP1);
    zVmax = _mm512_max_epi16(zVmax, zSrc1YP2);
    zVmin = _mm512_min_epi16(zVmin, zSrc1YP2);

    __m512i zSrc1XM2 = _mm512_alignr512_epi8<64-4>(zSrc1, zSrc0);
    __m512i zSrc1XM1 = _mm512_alignr512_epi8<64-2>(zSrc1, zSrc0);
    __m512i zSrc1XP1 = _mm512_alignr512_epi8<2>(zSrc2, zSrc1);
    __m512i zSrc1XP2 = _mm512_alignr512_epi8<4>(zSrc2, zSrc1);
    __m512i zMax  = _mm512_max_epi16(zSrc1, zSrc1XM2);
    __m512i zMin  = _mm512_min_epi16(zSrc1, zSrc1XM2);
    zMax  = _mm512_max_epi16(zMax, zSrc1XM1);
    zMin  = _mm512_min_epi16(zMin, zSrc1XM1);
    zMax  = _mm512_max_epi16(zMax, zSrc1XP1);
    zMin  = _mm512_min_epi16(zMin, zSrc1XP1);
    zMax  = _mm512_max_epi16(zMax, zSrc1XP2);
    zMin  = _mm512_min_epi16(zMin, zSrc1XP2);

    //if (max - min < vmax - vmin) { max = vmax, min = vmin; }
    __mmask32 kMask0 = _mm512_cmpgt_epi16_mask(_mm512_sub_epi16(zVmax, zVmin), _mm512_sub_epi16(zMax, zMin));
    zMax  = _mm512_mask_mov_epi16(zMax, kMask0, zVmax);
    zMin  = _mm512_mask_mov_epi16(zMin, kMask0, zVmin);

    //avg = (min + max) >> 1;
    __m512i zAvg = _mm512_srai_epi16(_mm512_add_epi16(zMax, zMin), 1);

    //if (max - min > thrs)
    __mmask32 kMask1 = _mm512_cmpgt_epi16_mask(_mm512_sub_epi16(zMax, zMin), zThreshold);

    //if (src->y == max) max += wc * 2;
    //else max += wc;
    __mmask32 kMask2 = _mm512_cmpeq_epi16_mask(zSrc1, zMax);
    zMax  = _mm512_add_epi16(zMax, zWc);
    zMax  = _mm512_mask_add_epi16(zMax, kMask2, zMax, zWc);

    //if (src->y == min) min -= bc * 2;
    //else  min -= bc;
    __mmask32 kMask3 = _mm512_cmpeq_epi16_mask(zSrc1, zMin);
    zMin  = _mm512_sub_epi16(zMin, zBc);
    zMin  = _mm512_mask_sub_epi16(zMin, kMask3, zMin, zBc);

    //dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
    __m512i z1, z0;
    z1    = _mm512_sub_epi16(zAvg, zSrc1);
    z0    = _mm512_unpacklo_epi16(z1, z1);
    z1    = _mm512_unpackhi_epi16(z1, z1);
    z0    = _mm512_madd_epi16(z0, zStrength);
    z1    = _mm512_madd_epi16(z1, zStrength);
    z0    = _mm512_srai_epi32(z0, 4);
    z1    = _mm512_srai_epi32(z1, 4);
    z0    = _mm512_packs_epi32(z0, z1);
    z0    = _mm512_add_epi16(z0, zSrc1);
    z0    = _mm512_max_epi16(z0, zMin);
    z0    = _mm512_min_epi16(z0, zMax);

    return _mm512_mask_mov_epi16(zSrc1, kMask1, z0);
}

template<bool avx512vbmi>
static __forceinline void multi_thread_func_avx512_line(BYTE *dst, BYTE *src, int w, int max_w,
    const __m512i& zThreshold, const __m512i& zStrength, const __m512i& zWc, const __m512i& zBc) {
    __m512i zSrc0 = get_previous_2_y_pixels(src);
    __m512i zSrc1 = load_y_from_yc48<avx512vbmi>(src);
    const BYTE *src_fin = src + w * PIXELYC_SIZE;
    for ( ; src < src_fin; src += 192, dst += 192) {
        //周辺近傍の最大と最小を縦方向・横方向に求める
        __m512i zSrc1YM2 = load_y_from_yc48<avx512vbmi>(src + (-2*max_w) * PIXELYC_SIZE);
        __m512i zSrc1YM1 = load_y_from_yc48<avx512vbmi>(src + (-1*max_w) * PIXELYC_SIZE);
        __m512i zSrc1YP1 = load_y_from_yc48<avx512vbmi>(src + ( 1*max_w) * PIXELYC_SIZE);
        __m512i zSrc1YP2 = load_y_from_yc48<avx512vbmi>(src + ( 2*max_w) * PIXELYC_SIZE);
        __m512i zSrc2    = load_y_from_yc48<avx512vbmi>(src +         32 * PIXELYC_SIZE);

        __m512i zY = edgelevel_avx512_32(zSrc1YM2, zSrc1YM1, zSrc0, zSrc1, zSrc2, zSrc1YP1, zSrc1YP2, zThreshold, zStrength, zWc, zBc);
        zSrc0 = zSrc1;
        zSrc1 = zSrc2;

        insert_y_yc48(dst, src, zY);
    }
}

void multi_thread_func_avx512(int thread_id, int thread_num, void *param1, void *param2) {
//  thread_id   : スレッド番号 ( 0 ～ thread_num-1 )
//  thread_num  : スレッド数 ( 1 ～ )
//  param1      : 汎用パラメータ
//  param2      : 汎用パラメータ
//
//  この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
    FILTER *fp              = (FILTER *)param1;
    FILTER_PROC_INFO *fpip  = (FILTER_PROC_INFO *)param2;

    const int max_w = fpip->max_w;
    const int h = fpip->h, w = fpip->w;
    const int str = fp->track[0], thrs = fp->track[1] << 3;
    const int bc = fp->track[2] << 4, wc = fp->track[3] << 4;
    const __m512i zStrength = _mm512_unpacklo_epi16(_mm512_setzero_si512(), _mm512_set1_epi16((short)(-1 * str)));
    const __m512i zThreshold = _mm512_set1_epi16((short)thrs);
    const __m512i zBc = _mm512_set1_epi16((short)bc);
    const __m512i zWc = _mm512_set1_epi16((short)wc);

    //  スレッド毎の画像を処理する場所を計算する
    const int y_start = (h *  thread_id   ) / thread_num;
    const int y_end   = (h * (thread_id+1)) / thread_num;

    BYTE *line_src = (BYTE *)fpip->ycp_edit + y_start * max_w * PIXELYC_SIZE;
    BYTE *line_dst = (BYTE *)fpip->ycp_temp + y_start * max_w * PIXELYC_SIZE;
    for (int y = y_start; y < y_end; y++, line_src += max_w * PIXELYC_SIZE, line_dst += max_w * PIXELYC_SIZE) {
        if (1 < y && y < h - 2) {
            int process_pixel = w;
            BYTE *src = line_src;
            BYTE *dst = line_dst;
            const int dst_mod64 = (size_t)dst & 0x3f;
            if (dst_mod64) {
                int mod6 = dst_mod64 % 6;
                int dw = (64 * (3 - (mod6 >> 1)) - dst_mod64) / 6;
                multi_thread_func_avx512_line<false>(dst, src, dw, max_w, zThreshold, zStrength, zWc, zBc);
                src += dw * PIXELYC_SIZE; dst += dw * PIXELYC_SIZE; process_pixel -= dw;
            }
            multi_thread_func_avx512_line<false>(dst, src, process_pixel, max_w, zThreshold, zStrength, zWc, zBc);

            __m64 m0 = *(__m64 *)(line_src + 0);
            DWORD d0 = *(DWORD *)(line_src + 8);
            *(__m64 *)(line_dst + 0) = m0;
            *(DWORD *)(line_dst + 8) = d0;
            __m128i x1 = _mm_loadu_si128((__m128i *)(line_src + w * PIXELYC_SIZE - 16));
            _mm_storeu_si128((__m128i *)(line_dst + w * PIXELYC_SIZE - 12), _mm_srli_si128(x1, 4));
        } else {
            memcpy_avx512<false>(line_dst, line_src, w * PIXELYC_SIZE);
        }
    }
    _mm256_zeroupper();
    _mm_empty();
}
