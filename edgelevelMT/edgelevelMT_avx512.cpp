#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "filter.h"
#include <cstdint>
#include <algorithm>
#include <immintrin.h>

#define PIXELYC_SIZE 6

#define BLOCK_OPT 0

#define SUFFLE_YCP_Y       _mm512_load_si512((const __m512i*)Array_SUFFLE_YCP_Y)
#define MASK_YCP_SELECT_Y  _mm512_load_si512((const __m512i*)Array_MASK_YCP_SELECT_Y)
#define MASK_FRAME_EDGE    _mm_load_si128((const __m128i*)Array_MASK_FRAME_EDGE)

//_mm512_srli_si512, _mm512_slli_si512は
//単に128bitシフト×2をするだけの命令である
#define _mm512_bsrli_epi128 _mm512_srli_si512
#define _mm512_bslli_epi128 _mm512_slli_si512

template<bool aligned_store>
static void __forceinline memcpy_avx512(void *_dst, const void *_src, int size) {
    char *dst = (char *)_dst;
    const char *src = (const char *)_src;
    if (size < 256) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    char *dst_fin = dst + size;
    char *dst_aligned_fin = (char *)(((size_t)(dst_fin + 63) & ~63) - 256);
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
    char *dst_tmp = dst_fin - 256;
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
__m512i __forceinline load_y_from_yc48(const char *src) {
    alignas(64) static const uint16_t PACK_YC48_SHUFFLE_AVX512[32] = {
         0,  3,  6,  9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
        48, 51, 54, 57, 60, 63,  2,  5,  8, 11, 14, 17, 20, 23, 26, 29
    };
    alignas(64) static const uint8_t PACK_YC48_SHUFFLE_AVX512_VBMI[64] = {
         0,   1,   6,   7,  12,  13,  18,  19,  24,  25,  30,  31, 36, 37, 42, 43, 48, 49, 54, 55, 60, 61, 66, 67, 72, 73, 78, 79, 84, 85, 90, 91,
        96,  97, 102, 103, 108, 109, 114, 115, 120, 121, 126, 127,  4,  5, 10, 11, 16, 17, 22, 23, 28, 29, 34, 35, 40, 41, 46, 47, 52, 53, 58, 59
    };
    __m512i z0 = _mm512_load_si512(avx512vbmi ? (const __m512i *)PACK_YC48_SHUFFLE_AVX512_VBMI : (const __m512i *)PACK_YC48_SHUFFLE_AVX512);
    __m512i z5 = _mm512_loadu_si512((const __m512i *)(src +   0));
    __m512i z4 = _mm512_loadu_si512((const __m512i *)(src +  64));
    __m512i z3 = _mm512_loadu_si512((const __m512i *)(src + 128));
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

static void __forceinline insert_y_yc48(char *dst, const char *src, const __m512i& zY) {
    alignas(64) static const uint16_t shuffle_yc48[] = {
         0, 11, 22,  1, 12, 23,  2, 13, 24,  3, 14, 25,  4, 15, 26,  5,
        16, 27,  6, 17, 28,  7, 18, 29,  8, 19, 30,  9, 20, 31, 10, 21

    };
    __m512i z0 = _mm512_loadu_si512((const __m512i *)(src +   0));
    __m512i z1 = _mm512_loadu_si512((const __m512i *)(src +  64));
    __m512i z2 = _mm512_loadu_si512((const __m512i *)(src + 128));

    __m512i z7 = _mm512_permutexvar_epi16(_mm512_loadu_si512((const __m512i *)(shuffle_yc48)), zY);

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

static void __forceinline insert_y_yc48(char *dst, const char *src, const __m512i &zY, int n) {
    alignas(64) static const uint16_t shuffle_yc48[] = {
         0, 11, 22,  1, 12, 23,  2, 13, 24,  3, 14, 25,  4, 15, 26,  5,
        16, 27,  6, 17, 28,  7, 18, 29,  8, 19, 30,  9, 20, 31, 10, 21

    };
    __m512i z0 = _mm512_loadu_si512((const __m512i *)(src + 0));
    __m512i z1 = _mm512_loadu_si512((const __m512i *)(src + 64));
    __m512i z2 = _mm512_loadu_si512((const __m512i *)(src + 128));

    __m512i z7 = _mm512_permutexvar_epi16(_mm512_loadu_si512((const __m512i *)(shuffle_yc48)), zY);

    __mmask32 k1 = 0x92492492u;
    __mmask32 k0 = k1 >> 1;
    __mmask32 k2 = k1 >> 2;
    z0 = _mm512_mask_mov_epi16(z0, k0, z7);
    z1 = _mm512_mask_mov_epi16(z1, k1, z7);
    z2 = _mm512_mask_mov_epi16(z2, k2, z7);

    n *= 3;

    __mmask32 kWrite;
    kWrite = (n >= 32) ? 0xffffffff : (1 << n) - 1;
    _mm512_mask_storeu_epi16((dst + 0), kWrite, z0);
    n = std::max(n-32,0);
    kWrite = (n >= 32) ? 0xffffffff : (1 << n) - 1;
    _mm512_mask_storeu_epi16((dst + 64), kWrite, z1);
    n = std::max(n - 32, 0);
    kWrite = (n >= 32) ? 0xffffffff : (1 << n) - 1;
    _mm512_mask_storeu_epi16((dst + 128), kWrite, z2);
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

static __forceinline __m512i get_previous_2_y_pixels(const char *src) {
    static const _declspec(align(16)) uint8_t SHUFFLE_LAST_2_Y_PIXELS[] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x04, 0x05, 0x0a, 0x0b
    };
    __m128i x0 = _mm_loadu_si128((const __m128i *)(src - 16));
    x0 = _mm_shuffle_epi8(x0, _mm_load_si128((const __m128i *)SHUFFLE_LAST_2_Y_PIXELS));
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

#if BLOCK_OPT
template<bool avx512vbmi, int line_size, bool fill_edge>
static __forceinline void fill_buffer(char *buf, const char *src, int x_start, int x_fin) {
    src += x_start * PIXELYC_SIZE;

    //自分の処理対象のブロックの左外側についてもロードしておく必要がある
    if (fill_edge) {
        //フレームの1行目、2行目については処理しない(コピーするだけ)なので、
        //左端で前方アクセスしても問題はない
        _mm512_store_si512((__m512i *)buf, get_previous_2_y_pixels(src));
    }

    const int count = x_fin - x_start;
    for (int x = 0; x < count; x += 32, buf += 64, src += 192) {
        _mm512_store_si512((__m512i *)(buf + 64), load_y_from_yc48<avx512vbmi>(src));
    }
    //自分の処理対象のブロックの右外側についてもロードしておく必要がある
    if (fill_edge) {
        _mm512_store_si512((__m512i *)(buf + 64), load_y_from_yc48<avx512vbmi>(src));
    }
}

#define BUFLINE(bufptr, y_line) ((char *)(bufptr) + (((y_line) & (buf_line - 1)) * line_size) * sizeof(int16_t))

template<bool avx512vbmi, int blocksize>
static void edgelevel_avx512_block(char *dst, const char *src, const int w, const int max_w, const int h,
    const int x_start, const int x_fin, int y_start, const int y_fin,
    const __m512i &zThreshold, const __m512i &zStrength, const __m512i &zWc, const __m512i &zBc) {
    //水平方向の加算結果を保持するバッファ
    const int buf_line = 4;
    const int line_size = blocksize + 32 * 2;
    int16_t __declspec(align(64)) buffer[buf_line * line_size];
    memset(buffer, 0, sizeof(buffer));

    const int src_pitch = max_w * PIXELYC_SIZE;
    const int dst_pitch = max_w * PIXELYC_SIZE;

    if (y_start == 0) {
        //フレームの1行目、2行目については処理しない(コピーするだけ)
        memcpy_avx512<false>(dst + 0 * dst_pitch + x_start * PIXELYC_SIZE, src + 0 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
        memcpy_avx512<false>(dst + 1 * dst_pitch + x_start * PIXELYC_SIZE, src + 1 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
        y_start += 2;
    }
    src += y_start * src_pitch;
    dst += y_start * dst_pitch;

    //計算に必要な情報をバッファにロードしておく
    for (int i = 0; i < 2; i++) {
        fill_buffer<avx512vbmi, line_size, false>(BUFLINE(buffer, i), src + (i - 2) * src_pitch, x_start, x_fin);
    }
    for (int i = 2; i < 4; i++) {
        fill_buffer<avx512vbmi, line_size, true>(BUFLINE(buffer, i), src + (i - 2) * src_pitch, x_start, x_fin);
    }

    int y = 0; //バッファのライン数のもととなるため、y=0で始めることは重要
    const int y_fin_loop = y_fin - y_start - ((y_fin >= h) ? 2 : 0); //フレームの最後の2行については処理しないのを考慮する
    for (; y < y_fin_loop; y++, dst += dst_pitch, src += src_pitch) {
        const char *src_ptr = src;
        char *dst_ptr = dst;
        char *buf_ptr = (char *)buffer;
        const int range_offset = src_pitch * 2;

        src_ptr += x_start * PIXELYC_SIZE;
        dst_ptr += x_start * PIXELYC_SIZE;

        __m512i zSrc0 = _mm512_loadu_si512((const __m512i *)(BUFLINE(buf_ptr, y+2) +  0));
        __m512i zSrc1 = _mm512_loadu_si512((const __m512i *)(BUFLINE(buf_ptr, y+2) + 64));

        //自分の処理対象のブロックの左外側についてもロードしておく必要がある
        _mm512_store_si512((__m512i *)BUFLINE(buf_ptr, y), get_previous_2_y_pixels(src_ptr + range_offset));

        int x = x_fin - x_start;
        for (; x > 32; x -= 32, buf_ptr += 64, src_ptr += 192, dst_ptr += 192) {
            //周辺近傍の最大と最小を縦方向・横方向に求める
            __m512i zSrc1YM2 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y+0) +  64));
            __m512i zSrc1YM1 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y+1) +  64));
            __m512i zSrc1YP1 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y+3) +  64));
            __m512i zSrc1YP2 = load_y_from_yc48<avx512vbmi>(src_ptr + range_offset);
            __m512i zSrc2    = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y+2) + 128));

            __m512i zY = edgelevel_avx512_32(zSrc1YM2, zSrc1YM1, zSrc0, zSrc1, zSrc2, zSrc1YP1, zSrc1YP2, zThreshold, zStrength, zWc, zBc);
            zSrc0 = zSrc1;
            zSrc1 = zSrc2;

            insert_y_yc48(dst_ptr, src_ptr, zY);
            _mm512_store_si512((__m512i *)(BUFLINE(buf_ptr, y+0) + 64), zSrc1YP2);
        }
        {   //最後のブロック
            //周辺近傍の最大と最小を縦方向・横方向に求める
            __m512i zSrc1YM2 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y + 0) + 64));
            __m512i zSrc1YM1 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y + 1) + 64));
            __m512i zSrc1YP1 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y + 3) + 64));
            __m512i zSrc1YP2 = load_y_from_yc48<avx512vbmi>(src_ptr + range_offset);
            __m512i zSrc2 = _mm512_load_si512((const __m512i *)(BUFLINE(buf_ptr, y + 2) + 128));

            __m512i zY = edgelevel_avx512_32(zSrc1YM2, zSrc1YM1, zSrc0, zSrc1, zSrc2, zSrc1YP1, zSrc1YP2, zThreshold, zStrength, zWc, zBc);

            insert_y_yc48(dst_ptr, src_ptr, zY, x);
            _mm512_store_si512((__m512i *)(BUFLINE(buf_ptr, y + 0) + 64), zSrc1YP2);
        }

        //自分の処理対象のブロックの右外側についてもロードしておく必要がある
        _mm512_store_si512((__m512i *)(BUFLINE(buf_ptr, y+0) + 128), load_y_from_yc48<avx512vbmi>(src_ptr + range_offset + 192));

        if (x_start == 0) {
            //左端2pixelはそのまま上書きコピーする
            *(uint32_t *)(dst + 0) = *(uint32_t *)(src + 0);
            *(uint32_t *)(dst + 4) = *(uint32_t *)(src + 4);
            *(uint32_t *)(dst + 8) = *(uint32_t *)(src + 8);
        }
        if (x_fin >= w) {
            //右端2pixelはそのまま上書きコピーする
            __m128i x1 = _mm_loadu_si128((__m128i *)(src + w * PIXELYC_SIZE - 16));
            _mm_storeu_si128((__m128i *)(dst + w * PIXELYC_SIZE - 12), _mm_srli_si128(x1, 4));
        }
    }
    if (y_fin >= h) {
        //フレームの最後の2行については処理しない(コピーするだけ)
        memcpy_avx512<false>(dst + 0 * dst_pitch + x_start * PIXELYC_SIZE, src + 0 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
        memcpy_avx512<false>(dst + 1 * dst_pitch + x_start * PIXELYC_SIZE, src + 1 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
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

    const char *ycp_src = (const char *)fpip->ycp_edit;
    char *ycp_dst = (char *)fpip->ycp_temp;

    //ブロックサイズの決定
    const int BLOCK_SIZE_YCP = 1024;
    const int max_block_size = BLOCK_SIZE_YCP;
    const int min_analyze_cycle = 64;
    const int scan_worker_x_limit_lower = std::min(thread_num, std::max(1, (w + BLOCK_SIZE_YCP - 1) / BLOCK_SIZE_YCP));
    const int scan_worker_x_limit_upper = std::max(1, w / 64);
    int scan_worker_x, scan_worker_y;
    for (int scan_worker_active = thread_num; ; scan_worker_active--) {
        for (scan_worker_x = scan_worker_x_limit_lower; scan_worker_x <= scan_worker_x_limit_upper; scan_worker_x++) {
            scan_worker_y = scan_worker_active / scan_worker_x;
            if (scan_worker_active - scan_worker_y * scan_worker_x == 0) {
                goto block_size_set; //二重ループを抜ける
            }
        }
    }
    block_size_set:
    if (thread_id >= scan_worker_x * scan_worker_y) {
        return;
    }
    int id_y = thread_id / scan_worker_x;
    int id_x = thread_id - id_y * scan_worker_x;
    int pos_y = ((int)(h * id_y / (double)scan_worker_y + 0.5)) & ~1;
    int y_fin = (id_y == scan_worker_y - 1) ? h : ((int)(h * (id_y+1) / (double)scan_worker_y + 0.5)) & ~1;
    int pos_x = ((int)(w * id_x / (double)scan_worker_x + 0.5) + (min_analyze_cycle -1)) & ~(min_analyze_cycle -1);
    int x_fin = (id_x == scan_worker_x - 1) ? w : ((int)(w * (id_x+1) / (double)scan_worker_x + 0.5) + (min_analyze_cycle -1)) & ~(min_analyze_cycle -1);
    if (pos_y == y_fin || pos_x == x_fin) {
        return; //念のため
    }
    int analyze_block = BLOCK_SIZE_YCP;
    if (id_x < scan_worker_x - 1) {
        for (; pos_x < x_fin; pos_x += analyze_block) {
            analyze_block = std::min(x_fin - pos_x, max_block_size);
            edgelevel_avx512_block<false, BLOCK_SIZE_YCP>(ycp_dst, ycp_src, w, max_w, h, pos_x, pos_x + analyze_block, pos_y, y_fin, zThreshold, zStrength, zWc, zBc);
        }
    } else {
        for (; x_fin - pos_x > max_block_size; pos_x += analyze_block) {
            analyze_block = std::min(x_fin - pos_x, max_block_size);
            edgelevel_avx512_block<false, BLOCK_SIZE_YCP>(ycp_dst, ycp_src, w, max_w, h, pos_x, pos_x + analyze_block, pos_y, y_fin, zThreshold, zStrength, zWc, zBc);
        }
        if (pos_x < w) {
            analyze_block = ((w - pos_x) + (min_analyze_cycle - 1)) & ~(min_analyze_cycle - 1);
            pos_x = w - analyze_block;
            edgelevel_avx512_block<false, BLOCK_SIZE_YCP>(ycp_dst, ycp_src, w, max_w, h, pos_x, pos_x + analyze_block, pos_y, y_fin, zThreshold, zStrength, zWc, zBc);
        }
    }
    _mm256_zeroupper();
}

#else
template<bool avx512vbmi>
static __forceinline void multi_thread_func_avx512_line(char *dst, const char *src, int w, int max_w,
    const __m512i& zThreshold, const __m512i& zStrength, const __m512i& zWc, const __m512i& zBc) {
    __m512i zSrc0 = get_previous_2_y_pixels(src);
    __m512i zSrc1 = load_y_from_yc48<avx512vbmi>(src);
    int x = w;
    for (; x > 32; x -= 32, src += 192, dst += 192) {
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

    __m512i zSrc1YM2 = load_y_from_yc48<avx512vbmi>(src + (-2*max_w) * PIXELYC_SIZE);
    __m512i zSrc1YM1 = load_y_from_yc48<avx512vbmi>(src + (-1*max_w) * PIXELYC_SIZE);
    __m512i zSrc1YP1 = load_y_from_yc48<avx512vbmi>(src + ( 1*max_w) * PIXELYC_SIZE);
    __m512i zSrc1YP2 = load_y_from_yc48<avx512vbmi>(src + ( 2*max_w) * PIXELYC_SIZE);
    __m512i zSrc2    = load_y_from_yc48<avx512vbmi>(src +         32 * PIXELYC_SIZE);

    __m512i zY = edgelevel_avx512_32(zSrc1YM2, zSrc1YM1, zSrc0, zSrc1, zSrc2, zSrc1YP1, zSrc1YP2, zThreshold, zStrength, zWc, zBc);

    insert_y_yc48(dst, src, zY, x);
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

    const char *line_src = (const char *)fpip->ycp_edit + y_start * max_w * PIXELYC_SIZE;
    char *line_dst = (char *)fpip->ycp_temp + y_start * max_w * PIXELYC_SIZE;
    for (int y = y_start; y < y_end; y++, line_src += max_w * PIXELYC_SIZE, line_dst += max_w * PIXELYC_SIZE) {
        if (1 < y && y < h - 2) {
            int process_pixel = w;
            const char *src = line_src;
            char *dst = line_dst;
            const int dst_mod64 = (size_t)dst & 0x3f;
            if (dst_mod64) {
                int mod6 = dst_mod64 % 6;
                int dw = (64 * (3 - (mod6 >> 1)) - dst_mod64) / 6;
                multi_thread_func_avx512_line<false>(dst, src, dw, max_w, zThreshold, zStrength, zWc, zBc);
                src += dw * PIXELYC_SIZE; dst += dw * PIXELYC_SIZE; process_pixel -= dw;
            }
            multi_thread_func_avx512_line<false>(dst, src, process_pixel, max_w, zThreshold, zStrength, zWc, zBc);

            *(uint32_t *)(line_dst + 0) = *(uint32_t *)(line_src + 0);
            *(uint32_t *)(line_dst + 4) = *(uint32_t *)(line_src + 4);
            *(uint32_t *)(line_dst + 8) = *(uint32_t *)(line_src + 8);
            dst = line_dst + w * PIXELYC_SIZE - 12;
            src = line_src + w * PIXELYC_SIZE - 12;
            *(uint32_t *)(dst + 0) = *(uint32_t *)(src + 0);
            *(uint32_t *)(dst + 4) = *(uint32_t *)(src + 4);
            *(uint32_t *)(dst + 8) = *(uint32_t *)(src + 8);
        } else {
            memcpy_avx512<false>(line_dst, line_src, w * PIXELYC_SIZE);
        }
    }
    _mm256_zeroupper();
}
#endif
