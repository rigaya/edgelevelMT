#pragma once

#include <Windows.h>
#include "filter.h"
#include <emmintrin.h> //SSE2
#if USE_SSSE3
#include <tmmintrin.h> //SSSE3
#endif
#if USE_SSE41
#include <smmintrin.h> //SSE4.1
#endif
#if USE_AVX
#include <immintrin.h> //AVX
#endif

#define PIXELYC_SIZE 6

#define ALIGN16_CONST_ARRAY static const _declspec(align(16))

ALIGN16_CONST_ARRAY BYTE   Array_SUFFLE_YCP_Y[]      = {0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11};
ALIGN16_CONST_ARRAY USHORT Array_MASK_YCP_SELECT_Y[] = {0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000};

#define SUFFLE_YCP_Y       _mm_load_si128((__m128i*)Array_SUFFLE_YCP_Y)
#define MASK_YCP_SELECT_Y  _mm_load_si128((__m128i*)Array_MASK_YCP_SELECT_Y)
#define MASK_FRAME_EDGE    _mm_load_si128((__m128i*)Array_MASK_FRAME_EDGE)

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です
template<bool dst_aligned, bool stream_store>
static void __forceinline memcpy_sse(BYTE *dst, const BYTE *src, int size) {
    if (size < 64) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    BYTE *dst_fin = dst + size;
    BYTE *dst_aligned_fin = (BYTE *)(((size_t)(dst_fin + 15) & ~15) - 64);
    __m128 x0, x1, x2, x3;
    if (!dst_aligned) {
        const int start_align_diff = (int)((size_t)dst & 15);
        if (start_align_diff) {
            x0 = _mm_loadu_ps((float*)src);
            _mm_storeu_ps((float*)dst, x0);
            dst += 16 - start_align_diff;
            src += 16 - start_align_diff;
        }
    }
#define _mm_store_switch_ps(ptr, xmm) ((stream_store) ? _mm_stream_ps((float *)(ptr), (xmm)) : _mm_store_ps((float *)(ptr), (xmm)))
    for ( ; dst < dst_aligned_fin; dst += 64, src += 64) {
        x0 = _mm_loadu_ps((float*)(src +  0));
        x1 = _mm_loadu_ps((float*)(src + 16));
        x2 = _mm_loadu_ps((float*)(src + 32));
        x3 = _mm_loadu_ps((float*)(src + 48));
        _mm_store_switch_ps((float*)(dst +  0), x0);
        _mm_store_switch_ps((float*)(dst + 16), x1);
        _mm_store_switch_ps((float*)(dst + 32), x2);
        _mm_store_switch_ps((float*)(dst + 48), x3);
    }
#undef _mm_store_switch_ps
    BYTE *dst_tmp = dst_fin - 64;
    src -= (dst - dst_tmp);
    x0 = _mm_loadu_ps((float*)(src +  0));
    x1 = _mm_loadu_ps((float*)(src + 16));
    x2 = _mm_loadu_ps((float*)(src + 32));
    x3 = _mm_loadu_ps((float*)(src + 48));
    _mm_storeu_ps((float*)(dst_tmp +  0), x0);
    _mm_storeu_ps((float*)(dst_tmp + 16), x1);
    _mm_storeu_ps((float*)(dst_tmp + 32), x2);
    _mm_storeu_ps((float*)(dst_tmp + 48), x3);
}
#pragma warning (pop)

static __forceinline __m128i blendv_epi8_simd(__m128i a, __m128i b, __m128i mask) {
#if USE_SSE41
    return _mm_blendv_epi8(a, b, mask);
#else
    return _mm_or_si128( _mm_andnot_si128(mask,a), _mm_and_si128(b,mask) );
#endif
}

//SSSE3のpalignrもどき
#define palignr_sse2(a,b,i) _mm_or_si128( _mm_slli_si128((a), (16-(i))), _mm_srli_si128((b), (i)) )

#if USE_SSSE3
#define _mm_alignr_epi8_simd _mm_alignr_epi8
#else
#define _mm_alignr_epi8_simd palignr_sse2
#endif

static inline __m128i shuffle_ycp_selected_y_sse2(__m128i x0) {
    x0 = _mm_shuffle_epi32(  x0, _MM_SHUFFLE(3,1,2,0));
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(3,0,2,1));
    x0 = _mm_shuffle_epi32(  x0, _MM_SHUFFLE(3,1,2,0));
    x0 = _mm_shufflehi_epi16(x0, _MM_SHUFFLE(1,2,3,0));
    x0 = _mm_shufflelo_epi16(x0, _MM_SHUFFLE(1,3,2,0));
    return x0;
}

template<bool aligned_load>
static __forceinline __m128i get_y_from_pixelyc(BYTE *src) {
#define _mm_switch_load_si128(x) ((aligned_load) ? _mm_load_si128((__m128i *)(x)) : _mm_loadu_si128((__m128i *)(x)))
    __m128i x0 = _mm_switch_load_si128((__m128i *)(src +  0));
    __m128i x1 = _mm_switch_load_si128((__m128i *)(src + 16));
    __m128i x2 = _mm_switch_load_si128((__m128i *)(src + 32));
#undef _mm_switch_load_si128
#if USE_SSE41
    const int MASK_INT = 0x40 + 0x08 + 0x01;
    x2 = _mm_blend_epi16(x2, x0, MASK_INT);
    x2 = _mm_blend_epi16(x2, x1, MASK_INT<<1);
#else
    x2 = blendv_epi8_simd(x2, x0, MASK_YCP_SELECT_Y);
    x2 = blendv_epi8_simd(x2, x1, _mm_slli_si128(MASK_YCP_SELECT_Y, 2));
#endif
#if USE_SSSE3
    x2 = _mm_shuffle_epi8(x2, SUFFLE_YCP_Y);
#else
    x2 = shuffle_ycp_selected_y_sse2(x2);
#endif
    return x2;
}

template<bool aligned_load>
static __forceinline void multi_thread_func_simd(int thread_id, int thread_num, void *param1, void *param2) {
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
    const __m128i xStrength = _mm_unpacklo_epi16(_mm_setzero_si128(), _mm_set1_epi16((short)(-1 * str)));
    const __m128i xThreshold = _mm_set1_epi16((short)thrs);
    const __m128i xBc = _mm_set1_epi16((short)bc);
    const __m128i xWc = _mm_set1_epi16((short)wc);
    //こんなにレジスタは多くないけど、適当に最適化してもらう
    __m128i x0, x1, x2, xSrc0, xSrc1, xSrc2, xTemp;
    __m128i xY, xVmin, xVmax, xMin, xMax, xMask, xAvg;

    //  スレッド毎の画像を処理する場所を計算する
    const int y_start = (h *  thread_id   ) / thread_num;
    const int y_end   = (h * (thread_id+1)) / thread_num;

    BYTE *line_src = (BYTE *)fpip->ycp_edit + y_start * max_w * PIXELYC_SIZE;
    BYTE *line_dst = (BYTE *)fpip->ycp_temp + y_start * max_w * PIXELYC_SIZE;
    for (int y = y_start; y < y_end; y++, line_src += max_w * PIXELYC_SIZE, line_dst += max_w * PIXELYC_SIZE) {
        if (1 < y && y < h - 2) {
            BYTE *src = line_src;
            BYTE *dst = line_dst;
            BYTE *src_fin = src + w * PIXELYC_SIZE;
            xSrc1 = get_y_from_pixelyc<aligned_load>(src);
            xSrc0 = _mm_shufflelo_epi16(xSrc1, 0);
            xSrc0 = _mm_shuffle_epi32(xSrc0, 0);
            for ( ; src < src_fin; src += 48, dst += 48) {
                //周辺近傍の最大と最小を縦方向・横方向に求める
                xVmax = get_y_from_pixelyc<aligned_load>(src + (-2*max_w) * PIXELYC_SIZE);
                xVmin = xVmax;
                xTemp = get_y_from_pixelyc<aligned_load>(src + (-1*max_w) * PIXELYC_SIZE);
                xVmax = _mm_max_epi16(xVmax, xTemp);
                xVmin = _mm_min_epi16(xVmin, xTemp);
                xSrc2 = get_y_from_pixelyc<aligned_load>(src + 8 * PIXELYC_SIZE);
                xMax  = xSrc1;
                xMin  = xMax;
                xTemp = _mm_alignr_epi8_simd(xSrc1, xSrc0, 16-4);
                xMax  = _mm_max_epi16(xMax, xTemp);
                xMin  = _mm_min_epi16(xMin, xTemp);
                xTemp = _mm_alignr_epi8_simd(xSrc1, xSrc0, 16-2);
                xMax  = _mm_max_epi16(xMax, xTemp);
                xMin  = _mm_min_epi16(xMin, xTemp);
                xVmax = _mm_max_epi16(xVmax, xSrc1);
                xVmin = _mm_min_epi16(xVmin, xSrc1);
                xTemp = _mm_alignr_epi8_simd(xSrc2, xSrc1, 2);
                xMax  = _mm_max_epi16(xMax, xTemp);
                xMin  = _mm_min_epi16(xMin, xTemp);
                xTemp = _mm_alignr_epi8_simd(xSrc2, xSrc1, 4);
                xMax  = _mm_max_epi16(xMax, xTemp);
                xMin  = _mm_min_epi16(xMin, xTemp);
                xTemp = get_y_from_pixelyc<aligned_load>(src + (1*max_w) * PIXELYC_SIZE);
                xVmax = _mm_max_epi16(xVmax, xTemp);
                xVmin = _mm_min_epi16(xVmin, xTemp);
                xTemp = get_y_from_pixelyc<aligned_load>(src + (2*max_w) * PIXELYC_SIZE);
                xVmax = _mm_max_epi16(xVmax, xTemp);
                xVmin = _mm_min_epi16(xVmin, xTemp);
                xSrc0 = xSrc1;
                xSrc1 = xSrc2;

                //if (max - min < vmax - vmin) { max = vmax, min = vmin; }
                xMask = _mm_cmpgt_epi16(_mm_sub_epi16(xVmax, xVmin), _mm_sub_epi16(xMax, xMin));
                xMax  = blendv_epi8_simd(xMax, xVmax, xMask);
                xMin  = blendv_epi8_simd(xMin, xVmin, xMask);

                //avg = (min + max) >> 1;
                xAvg  = _mm_add_epi16(xMax, xMin);
                xAvg  = _mm_srai_epi16(xAvg, 1);

                //if (max - min > thrs)
                xMask = _mm_cmpgt_epi16(_mm_sub_epi16(xMax, xMin), xThreshold);

                //if (src->y == max) max += wc * 2;
                //else max += wc;
                x1    = _mm_cmpeq_epi16(xSrc0, xMax);
                xMax  = _mm_add_epi16(xMax, xWc);
                xMax  = _mm_add_epi16(xMax, _mm_and_si128(xWc, x1));

                //if (src->y == min) min -= bc * 2;
                //else  min -= bc;
                x1    = _mm_cmpeq_epi16(xSrc0, xMin);
                xMin  = _mm_sub_epi16(xMin, xBc);
                xMin  = _mm_sub_epi16(xMin, _mm_and_si128(xBc, x1));

                //dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
                x1    = _mm_sub_epi16(xAvg, xSrc0);
                x0    = _mm_unpacklo_epi16(x1, x1);
                x1    = _mm_unpackhi_epi16(x1, x1);
                x0    = _mm_madd_epi16(x0, xStrength);
                x1    = _mm_madd_epi16(x1, xStrength);
                x0    = _mm_srai_epi32(x0, 4);
                x1    = _mm_srai_epi32(x1, 4);
                x0    = _mm_packs_epi32(x0, x1);
                x0    = _mm_add_epi16(x0, xSrc0);
                x0    = _mm_max_epi16(x0, xMin);
                x0    = _mm_min_epi16(x0, xMax);

                xY    = blendv_epi8_simd(xSrc0, x0, xMask);

                x0 = _mm_loadu_si128((__m128i *)(src +  0));
                x1 = _mm_loadu_si128((__m128i *)(src + 16));
                x2 = _mm_loadu_si128((__m128i *)(src + 32));
#if USE_SSE41
                const int MASK_INT = 0x40 + 0x08 + 0x01;
                xY = _mm_shuffle_epi8(xY, SUFFLE_YCP_Y);
                x0 = _mm_blend_epi16(x0, xY, MASK_INT);
                x1 = _mm_blend_epi16(x1, xY, MASK_INT<<1);
                x2 = _mm_blend_epi16(x2, xY, (MASK_INT<<2) & 0xFF);
#elif USE_SSSE3
                xY = _mm_shuffle_epi8(xY, SUFFLE_YCP_Y);
                x0 = blendv_epi8_simd(x0, xY, MASK_YCP_SELECT_Y);
                x1 = blendv_epi8_simd(x1, xY, _mm_slli_si128(MASK_YCP_SELECT_Y, 2));
                x2 = blendv_epi8_simd(x2, xY, _mm_slli_si128(MASK_YCP_SELECT_Y, 4));
#else
                xY = shuffle_ycp_selected_y_sse2(xY);
                x0 = blendv_epi8_simd(x0, xY, MASK_YCP_SELECT_Y);
                x1 = blendv_epi8_simd(x1, xY, _mm_slli_si128(MASK_YCP_SELECT_Y, 2));
                x2 = blendv_epi8_simd(x2, xY, _mm_slli_si128(MASK_YCP_SELECT_Y, 4));
#endif
                _mm_storeu_si128((__m128i *)(dst +  0), x0);
                _mm_storeu_si128((__m128i *)(dst + 16), x1);
                _mm_storeu_si128((__m128i *)(dst + 32), x2);
            }
            __m64 m0 = *(__m64 *)(line_src + 0);
            DWORD d0 = *(DWORD *)(line_src + 8);
            *(__m64 *)(line_dst + 0) = m0;
            *(DWORD *)(line_dst + 8) = d0;
            x1 = _mm_loadu_si128((__m128i *)(line_src + w * PIXELYC_SIZE - 16));
            _mm_storeu_si128((__m128i *)(line_dst + w * PIXELYC_SIZE - 12), _mm_srli_si128(x1, 4));
        } else {
            memcpy_sse<false, false>(line_dst, line_src, w * PIXELYC_SIZE);
        }
    }
    _mm_empty();
}
