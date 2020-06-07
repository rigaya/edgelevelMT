#pragma once

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include "filter.h"
#include <immintrin.h> //AVX, AVX2
#include <algorithm>

#define BLOCK_OPT 0

#define PIXELYC_SIZE 6

#define ALIGN32_CONST_ARRAY static const _declspec(align(32))

ALIGN32_CONST_ARRAY BYTE   Array_SUFFLE_YCP_Y[]      = {0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11};
ALIGN32_CONST_ARRAY USHORT Array_MASK_YCP_SELECT_Y[] = {0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000};

#define SUFFLE_YCP_Y       _mm256_load_si256((const __m256i*)Array_SUFFLE_YCP_Y)
#define MASK_YCP_SELECT_Y  _mm256_load_si256((const __m256i*)Array_MASK_YCP_SELECT_Y)
#define MASK_FRAME_EDGE    _mm_load_si128((const __m128i*)Array_MASK_FRAME_EDGE)

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です
template<bool dst_aligned, bool use_stream, bool zeroupper>
static void __forceinline memcpy_avx2(char *dst, const char *src, int size) {
    if (size < 128) {
        for (int i = 0; i < size; i++)
            dst[i] = src[i];
        return;
    }
    char *dst_fin = dst + size;
    char *dst_aligned_fin = (char *)(((size_t)(dst_fin + 31) & ~31) - 128);
    __m256i y0, y1, y2, y3;
    if (!dst_aligned) {
        const int start_align_diff = (int)((size_t)dst & 31);
        if (start_align_diff) {
            y0 = _mm256_loadu_si256((__m256i*)src);
            _mm256_storeu_si256((__m256i*)dst, y0);
            dst += 32 - start_align_diff;
            src += 32 - start_align_diff;
        }
    }
#define _mm256_stream_switch_si256(x, ymm) ((use_stream) ? _mm256_stream_si256((x), (ymm)) : _mm256_store_si256((x), (ymm)))
    for ( ; dst < dst_aligned_fin; dst += 128, src += 128) {
        y0 = _mm256_loadu_si256((const __m256i*)(src +  0));
        y1 = _mm256_loadu_si256((const __m256i*)(src + 32));
        y2 = _mm256_loadu_si256((const __m256i*)(src + 64));
        y3 = _mm256_loadu_si256((const __m256i*)(src + 96));
        _mm256_stream_switch_si256((__m256i*)(dst +  0), y0);
        _mm256_stream_switch_si256((__m256i*)(dst + 32), y1);
        _mm256_stream_switch_si256((__m256i*)(dst + 64), y2);
        _mm256_stream_switch_si256((__m256i*)(dst + 96), y3);
    }
#undef _mm256_stream_switch_si256
    char *dst_tmp = dst_fin - 128;
    src -= (dst - dst_tmp);
    y0 = _mm256_loadu_si256((const __m256i*)(src +  0));
    y1 = _mm256_loadu_si256((const __m256i*)(src + 32));
    y2 = _mm256_loadu_si256((const __m256i*)(src + 64));
    y3 = _mm256_loadu_si256((const __m256i*)(src + 96));
    _mm256_storeu_si256((__m256i*)(dst_tmp +  0), y0);
    _mm256_storeu_si256((__m256i*)(dst_tmp + 32), y1);
    _mm256_storeu_si256((__m256i*)(dst_tmp + 64), y2);
    _mm256_storeu_si256((__m256i*)(dst_tmp + 96), y3);
    if (zeroupper) {
        _mm256_zeroupper();
    }
}
#pragma warning (pop)

//本来の256bit alignr
#define MM_ABS(x) (((x) < 0) ? -(x) : (x))
#define _mm256_alignr256_epi8(a, b, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), b, i) : _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, b, (0x00<<4) + 0x03), MM_ABS(i-16)))

//_mm256_srli_si256, _mm256_slli_si256は
//単に128bitシフト×2をするだけの命令である
#define _mm256_bsrli_epi128 _mm256_srli_si256
#define _mm256_bslli_epi128 _mm256_slli_si256

//本当の256bitシフト
#define _mm256_srli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), a, i) : _mm256_bsrli_epi128(_mm256_permute2x128_si256(a, a, (0x08<<4) + 0x03), MM_ABS(i-16)))
#define _mm256_slli256_si256(a, i) ((i<=16) ? _mm256_alignr_epi8(a, _mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(16-i)) : _mm256_bslli_epi128(_mm256_permute2x128_si256(a, a, (0x00<<4) + 0x08), MM_ABS(i-16)))


static __forceinline __m256i get_y_from_pixelyc(const char *src) {
    __m256i y0 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 48)), _mm_loadu_si128((const __m128i *)(src +  0)));
    __m256i y1 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 64)), _mm_loadu_si128((const __m128i *)(src + 16)));
    __m256i y2 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 80)), _mm_loadu_si128((const __m128i *)(src + 32)));
    const int MASK_INT = 0x40 + 0x08 + 0x01;
    y2 = _mm256_blend_epi16(y2, y0, MASK_INT);
    y2 = _mm256_blend_epi16(y2, y1, MASK_INT<<1);
    y2 = _mm256_shuffle_epi8(y2, SUFFLE_YCP_Y);
    return y2;
}

template<bool aligned_store>
static void __forceinline insert_y_yc48(char *dst, const char *src, __m256i yY) {

    __m256i y0 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 48)), _mm_loadu_si128((const __m128i *)(src +  0))); // 384,   0
    __m256i y1 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 64)), _mm_loadu_si128((const __m128i *)(src + 16))); // 512, 128
    __m256i y2 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src + 80)), _mm_loadu_si128((const __m128i *)(src + 32))); // 768, 256

    const int MASK_INT = 0x40 + 0x08 + 0x01;
    yY = _mm256_shuffle_epi8(yY, SUFFLE_YCP_Y);
    y0 = _mm256_blend_epi16(y0, yY, MASK_INT);
    y1 = _mm256_blend_epi16(y1, yY, MASK_INT<<1);
    y2 = _mm256_blend_epi16(y2, yY, (MASK_INT<<2) & 0xFF);

#define _mm256_store_switch_si256(ptr, xmm) ((aligned_store) ? _mm256_stream_si256((__m256i *)(ptr), (xmm)) : _mm256_storeu_si256((__m256i *)(ptr), (xmm)))
    _mm256_store_switch_si256((__m256i *)(dst +  0), _mm256_permute2x128_si256(y0, y1, (0x02<<4)+0x00)); // 128,   0
    _mm256_store_switch_si256((__m256i *)(dst + 32), _mm256_blend_epi32(       y0, y2, (0x00<<4)+0x0f)); // 384, 256
    _mm256_store_switch_si256((__m256i *)(dst + 64), _mm256_permute2x128_si256(y1, y2, (0x03<<4)+0x01)); // 768, 512
#undef _mm256_store_switch_si256
}

static __forceinline __m256i get_previous_2_y_pixels(const char *src) {
    static const _declspec(align(32)) uint8_t SHUFFLE_LAST_2_Y_PIXELS[] = {
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x04, 0x05, 0x0a, 0x0b
    };
    __m256i y0 = _mm256_set_m128i(_mm_loadu_si128((const __m128i *)(src - 16)), _mm_setzero_si128());
    y0 = _mm256_shuffle_epi8(y0, _mm256_load_si256((const __m256i*)SHUFFLE_LAST_2_Y_PIXELS));
    return y0;
}

static __forceinline __m256i edgelevel_avx2_16(
    const __m256i &ySrc1YM2, const __m256i &ySrc1YM1,
    const __m256i &ySrc0, const __m256i &ySrc1, const __m256i &ySrc2,
    const __m256i &ySrc1YP1, const __m256i &ySrc1YP2,
    const __m256i &yThreshold, const __m256i &yStrength, const __m256i &yWc, const __m256i &yBc) {
    //周辺近傍の最大と最小を縦方向・横方向に求める
    __m256i yVmax, yVmin;
    yVmax = _mm256_max_epi16(ySrc1, ySrc1YM2);
    yVmin = _mm256_min_epi16(ySrc1, ySrc1YM2);
    yVmax = _mm256_max_epi16(yVmax, ySrc1YM1);
    yVmin = _mm256_min_epi16(yVmin, ySrc1YM1);
    yVmax = _mm256_max_epi16(yVmax, ySrc1YP1);
    yVmin = _mm256_min_epi16(yVmin, ySrc1YP1);
    yVmax = _mm256_max_epi16(yVmax, ySrc1YP2);
    yVmin = _mm256_min_epi16(yVmin, ySrc1YP2);

    __m256i ySrc1XM2 = _mm256_alignr256_epi8(ySrc1, ySrc0, 32-4);
    __m256i ySrc1XM1 = _mm256_alignr256_epi8(ySrc1, ySrc0, 32-2);
    __m256i ySrc1XP1 = _mm256_alignr256_epi8(ySrc2, ySrc1, 2);
    __m256i ySrc1XP2 = _mm256_alignr256_epi8(ySrc2, ySrc1, 4);
    __m256i yMax, yMin;
    yMax  = _mm256_max_epi16(ySrc1, ySrc1XM2);
    yMin  = _mm256_min_epi16(ySrc1, ySrc1XM2);
    yMax  = _mm256_max_epi16(yMax, ySrc1XM1);
    yMin  = _mm256_min_epi16(yMin, ySrc1XM1);
    yMax  = _mm256_max_epi16(yMax, ySrc1XP1);
    yMin  = _mm256_min_epi16(yMin, ySrc1XP1);
    yMax  = _mm256_max_epi16(yMax, ySrc1XP2);
    yMin  = _mm256_min_epi16(yMin, ySrc1XP2);

    //if (max - min < vmax - vmin) { max = vmax, min = vmin; }
    __m256i yMask = _mm256_cmpgt_epi16(_mm256_sub_epi16(yVmax, yVmin), _mm256_sub_epi16(yMax, yMin));
    yMax  = _mm256_blendv_epi8(yMax, yVmax, yMask);
    yMin  = _mm256_blendv_epi8(yMin, yVmin, yMask);

    //avg = (min + max) >> 1;
    __m256i yAvg = _mm256_srai_epi16(_mm256_add_epi16(yMax, yMin), 1);

    //if (max - min > thrs)
    yMask = _mm256_cmpgt_epi16(_mm256_sub_epi16(yMax, yMin), yThreshold);

    //if (src->y == max) max += wc * 2;
    //else max += wc;
    __m256i yMaskMax = _mm256_cmpeq_epi16(ySrc1, yMax);
    yMax  = _mm256_add_epi16(yMax, yWc);
    yMax  = _mm256_add_epi16(yMax, _mm256_and_si256(yWc, yMaskMax));

    //if (src->y == min) min -= bc * 2;
    //else  min -= bc;
    __m256i yMaskMin = _mm256_cmpeq_epi16(ySrc1, yMin);
    yMin  = _mm256_sub_epi16(yMin, yBc);
    yMin  = _mm256_sub_epi16(yMin, _mm256_and_si256(yBc, yMaskMin));

    //dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
    __m256i y0, y1;
    y1    = _mm256_sub_epi16(yAvg, ySrc1);
    y0    = _mm256_unpacklo_epi16(y1, y1);
    y1    = _mm256_unpackhi_epi16(y1, y1);
    y0    = _mm256_madd_epi16(y0, yStrength);
    y1    = _mm256_madd_epi16(y1, yStrength);
    y0    = _mm256_srai_epi32(y0, 4);
    y1    = _mm256_srai_epi32(y1, 4);
    y0    = _mm256_packs_epi32(y0, y1);
    y0    = _mm256_add_epi16(y0, ySrc1);
    y0    = _mm256_max_epi16(y0, yMin);
    y0    = _mm256_min_epi16(y0, yMax);

    return _mm256_blendv_epi8(ySrc1, y0, yMask);
}


#if BLOCK_OPT

template<int line_size, bool fill_edge>
static __forceinline void fill_buffer(char *buf, const char *src, int x_start, int x_fin) {
    src += x_start * PIXELYC_SIZE;

    //自分の処理対象のブロックの左外側についてもロードしておく必要がある
    if (fill_edge) {
        //フレームの1行目、2行目については処理しない(コピーするだけ)なので、
        //左端で前方アクセスしても問題はない
        _mm256_store_si256((__m256i *)buf, get_previous_2_y_pixels(src));
    }

    const int count = x_fin - x_start;
    for (int x = 0; x < count; x += 16, buf += 32, src += 96) {
        _mm256_store_si256((__m256i *)(buf + 32), get_y_from_pixelyc(src));
    }
    //自分の処理対象のブロックの右外側についてもロードしておく必要がある
    if (fill_edge) {
        _mm256_store_si256((__m256i *)(buf + 32), get_y_from_pixelyc(src));
    }
}

#define BUFLINE(bufptr, y_line) ((char *)(bufptr) + (((y_line) & (buf_line - 1)) * line_size) * sizeof(int16_t))

template<int blocksize>
static void edgelevel_avx2_block(char *dst, const char *src, const int w, const int max_w, const int h,
    const int x_start, const int x_fin, int y_start, const int y_fin,
    const __m256i &yThreshold, const __m256i &yStrength, const __m256i &yWc, const __m256i &yBc) {
    //水平方向の加算結果を保持するバッファ
    const int buf_line = 4;
    const int line_size = blocksize + 16 * 2;
    int16_t __declspec(align(32)) buffer[buf_line * line_size];
    memset(buffer, 0, sizeof(buffer));

    const int src_pitch = max_w * PIXELYC_SIZE;
    const int dst_pitch = max_w * PIXELYC_SIZE;

    if (y_start == 0) {
        //フレームの1行目、2行目については処理しない(コピーするだけ)
        memcpy_avx2<false, false, false>(dst + 0 * dst_pitch + x_start * PIXELYC_SIZE, src + 0 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
        memcpy_avx2<false, false, false>(dst + 1 * dst_pitch + x_start * PIXELYC_SIZE, src + 1 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
        y_start += 2;
    }
    src += y_start * src_pitch;
    dst += y_start * dst_pitch;

    //計算に必要な情報をバッファにロードしておく
    for (int i = 0; i < 2; i++) {
        fill_buffer<line_size, false>(BUFLINE(buffer, i), src + (i - 2) * src_pitch, x_start, x_fin);
    }
    for (int i = 2; i < 4; i++) {
        fill_buffer<line_size, true>(BUFLINE(buffer, i), src + (i - 2) * src_pitch, x_start, x_fin);
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

        __m256i ySrc0 = _mm256_loadu_si256((const __m256i *)(BUFLINE(buf_ptr, y+2) +  0));
        __m256i ySrc1 = _mm256_loadu_si256((const __m256i *)(BUFLINE(buf_ptr, y+2) + 32));

        //自分の処理対象のブロックの左外側についてもロードしておく必要がある
        _mm256_store_si256((__m256i *)BUFLINE(buf_ptr, y), get_previous_2_y_pixels(src_ptr + range_offset));

        const int count = x_fin - x_start;
        for (int x = 0; x < count; x += 16, buf_ptr += 32, src_ptr += 96, dst_ptr += 96) {
            //周辺近傍の最大と最小を縦方向・横方向に求める
            __m256i ySrc1YM2 = _mm256_load_si256((const __m256i *)(BUFLINE(buf_ptr, y+0) + 32));
            __m256i ySrc1YM1 = _mm256_load_si256((const __m256i *)(BUFLINE(buf_ptr, y+1) + 32));
            __m256i ySrc1YP1 = _mm256_load_si256((const __m256i *)(BUFLINE(buf_ptr, y+3) + 32));
            __m256i ySrc1YP2 = get_y_from_pixelyc(src_ptr + range_offset);
            __m256i ySrc2    = _mm256_load_si256((const __m256i *)(BUFLINE(buf_ptr, y+2) + 64));

            __m256i yY = edgelevel_avx2_16(ySrc1YM2, ySrc1YM1, ySrc0, ySrc1, ySrc2, ySrc1YP1, ySrc1YP2, yThreshold, yStrength, yWc, yBc);
            ySrc0 = ySrc1;
            ySrc1 = ySrc2;

            insert_y_yc48<false>(dst_ptr, src_ptr, yY);
            _mm256_store_si256((__m256i *)(BUFLINE(buf_ptr, y+0) + 32), ySrc1YP2);
        }
        //自分の処理対象のブロックの右外側についてもロードしておく必要がある
        _mm256_store_si256((__m256i *)(BUFLINE(buf_ptr, y+0) + 32), get_y_from_pixelyc(src_ptr + range_offset));

        if (x_start == 0) {
            //左端2pixelはそのまま上書きコピーする
            *(DWORD *)(dst + 0) = *(DWORD *)(src + 0);
            *(DWORD *)(dst + 4) = *(DWORD *)(src + 4);
            *(DWORD *)(dst + 8) = *(DWORD *)(src + 8);
        }
        if (x_fin >= w) {
            //右端2pixelはそのまま上書きコピーする
            __m128i x1 = _mm_loadu_si128((__m128i *)(src + w * PIXELYC_SIZE - 16));
            _mm_storeu_si128((__m128i *)(dst + w * PIXELYC_SIZE - 12), _mm_srli_si128(x1, 4));
        }
    }
    if (y_fin >= h) {
        //フレームの最後の2行については処理しない(コピーするだけ)
        memcpy_avx2<false, false, false>(dst + 0 * dst_pitch + x_start * PIXELYC_SIZE, src + 0 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
        memcpy_avx2<false, false, false>(dst + 1 * dst_pitch + x_start * PIXELYC_SIZE, src + 1 * src_pitch + x_start * PIXELYC_SIZE, (x_fin - x_start) * PIXELYC_SIZE);
    }
}

void multi_thread_func_avx2(int thread_id, int thread_num, void *param1, void *param2) {
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
    const __m256i yStrength = _mm256_unpacklo_epi16(_mm256_setzero_si256(), _mm256_set1_epi16((short)(-1 * str)));
    const __m256i yThreshold = _mm256_set1_epi16((short)thrs);
    const __m256i yBc = _mm256_set1_epi16((short)bc);
    const __m256i yWc = _mm256_set1_epi16((short)wc);


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
            edgelevel_avx2_block<BLOCK_SIZE_YCP>(ycp_dst, ycp_src, w, max_w, h, pos_x, pos_x + analyze_block, pos_y, y_fin, yThreshold, yStrength, yWc, yBc);
        }
    } else {
        for (; x_fin - pos_x > max_block_size; pos_x += analyze_block) {
            analyze_block = std::min(x_fin - pos_x, max_block_size);
            edgelevel_avx2_block<BLOCK_SIZE_YCP>(ycp_dst, ycp_src, w, max_w, h, pos_x, pos_x + analyze_block, pos_y, y_fin, yThreshold, yStrength, yWc, yBc);
        }
        if (pos_x < w) {
            analyze_block = ((w - pos_x) + (min_analyze_cycle - 1)) & ~(min_analyze_cycle - 1);
            pos_x = w - analyze_block;
            edgelevel_avx2_block<BLOCK_SIZE_YCP>(ycp_dst, ycp_src, w, max_w, h, pos_x, pos_x + analyze_block, pos_y, y_fin, yThreshold, yStrength, yWc, yBc);
        }
    }
}
#else
template<bool aligned_store>
static __forceinline void multi_thread_func_avx2_line(char *dst, const char *src, int w, int max_w,
    const __m256i& yThreshold, const __m256i& yStrength, const __m256i& yWc, const __m256i& yBc) {
    __m256i ySrc0 = get_previous_2_y_pixels(src);
    __m256i ySrc1 = get_y_from_pixelyc(src);
    const char *src_fin = src + w * PIXELYC_SIZE;
    for ( ; src < src_fin; src += 96, dst += 96) {
        //周辺近傍の最大と最小を縦方向・横方向に求める
        __m256i ySrc1YM2 = get_y_from_pixelyc(src + (-2*max_w) * PIXELYC_SIZE);
        __m256i ySrc1YM1 = get_y_from_pixelyc(src + (-1*max_w) * PIXELYC_SIZE);
        __m256i ySrc1YP1 = get_y_from_pixelyc(src + ( 1*max_w) * PIXELYC_SIZE);
        __m256i ySrc1YP2 = get_y_from_pixelyc(src + ( 2*max_w) * PIXELYC_SIZE);
        __m256i ySrc2    = get_y_from_pixelyc(src +         16 * PIXELYC_SIZE);

        __m256i yY = edgelevel_avx2_16(ySrc1YM2, ySrc1YM1, ySrc0, ySrc1, ySrc2, ySrc1YP1, ySrc1YP2, yThreshold, yStrength, yWc, yBc);
        ySrc0 = ySrc1;
        ySrc1 = ySrc2;

        insert_y_yc48<aligned_store>(dst, src, yY);
    }
}

void multi_thread_func_avx2(int thread_id, int thread_num, void *param1, void *param2) {
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
    const __m256i yStrength = _mm256_unpacklo_epi16(_mm256_setzero_si256(), _mm256_set1_epi16((short)(-1 * str)));
    const __m256i yThreshold = _mm256_set1_epi16((short)thrs);
    const __m256i yBc = _mm256_set1_epi16((short)bc);
    const __m256i yWc = _mm256_set1_epi16((short)wc);

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
            const int dst_mod32 = (size_t)dst & 0x1f;
            if (dst_mod32) {
                int mod6 = dst_mod32 % 6;
                int dw = (32 * (((mod6) ? mod6 : 6)>>1)-dst_mod32) / 6;
                multi_thread_func_avx2_line<false>(dst, src, dw, max_w, yThreshold, yStrength, yWc, yBc);
                src += dw * PIXELYC_SIZE; dst += dw * PIXELYC_SIZE; process_pixel -= dw;
            }
            multi_thread_func_avx2_line<true>(dst, src, process_pixel, max_w, yThreshold, yStrength, yWc, yBc);

            __m64 m0 = *(__m64 *)(line_src + 0);
            DWORD d0 = *(DWORD *)(line_src + 8);
            *(__m64 *)(line_dst + 0) = m0;
            *(DWORD *)(line_dst + 8) = d0;
            __m128i x1 = _mm_loadu_si128((__m128i *)(line_src + w * PIXELYC_SIZE - 16));
            _mm_storeu_si128((__m128i *)(line_dst + w * PIXELYC_SIZE - 12), _mm_srli_si128(x1, 4));
        } else {
            memcpy_avx2<false, false, false>(line_dst, line_src, w * PIXELYC_SIZE);
        }
    }
    _mm256_zeroupper();
    _mm_empty();
}
#endif