#pragma once

#include <Windows.h>
#include "filter.h"
#include <immintrin.h> //AVX, AVX2

#define PIXELYC_SIZE 6

#define ALIGN32_CONST_ARRAY static const _declspec(align(32))

ALIGN32_CONST_ARRAY BYTE   Array_SUFFLE_YCP_Y[]      = {0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11};
ALIGN32_CONST_ARRAY USHORT Array_MASK_YCP_SELECT_Y[] = {0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000};

#define SUFFLE_YCP_Y       _mm256_load_si256((__m256i*)Array_SUFFLE_YCP_Y)
#define MASK_YCP_SELECT_Y  _mm256_load_si256((__m256i*)Array_MASK_YCP_SELECT_Y)
#define MASK_FRAME_EDGE    _mm_load_si128((__m128i*)Array_MASK_FRAME_EDGE)

#pragma warning (push)
#pragma warning (disable: 4127) //warning C4127: 条件式が定数です
template<bool dst_aligned, bool use_stream, bool zeroupper>
static void __forceinline memcpy_avx2(BYTE *dst, const BYTE *src, int size) {
	if (size < 128) {
		for (int i = 0; i < size; i++)
			dst[i] = src[i];
		return;
	}
	BYTE *dst_fin = dst + size;
	BYTE *dst_aligned_fin = (BYTE *)(((size_t)(dst_fin + 31) & ~31) - 128);
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
	BYTE *dst_tmp = dst_fin - 128;
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


static __forceinline __m256i get_y_from_pixelyc(BYTE *src) {
	__m256i y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src + 48)), _mm_loadu_si128((__m128i *)(src +  0)));
	__m256i y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src + 64)), _mm_loadu_si128((__m128i *)(src + 16)));
	__m256i y2 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src + 80)), _mm_loadu_si128((__m128i *)(src + 32)));
	const int MASK_INT = 0x40 + 0x08 + 0x01;
	y2 = _mm256_blend_epi16(y2, y0, MASK_INT);
	y2 = _mm256_blend_epi16(y2, y1, MASK_INT<<1);
	y2 = _mm256_shuffle_epi8(y2, SUFFLE_YCP_Y);
	return y2;
}

static __forceinline __m256i get_previous_2_y_pixels(BYTE *src) {
	static const _declspec(align(32)) BYTE SHUFFLE_LAST_2_Y_PIXELS[] = {
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x04, 0x05, 0x0a, 0x0b
	};
	__m256i y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src - 16)), _mm_setzero_si128());
	y0 = _mm256_shuffle_epi8(y0, _mm256_load_si256((__m256i*)SHUFFLE_LAST_2_Y_PIXELS));
	return y0;
}

template<bool aligned_store>
static __forceinline void multi_thread_func_avx2_line(BYTE *dst, BYTE *src, int w, int max_w,
	const __m256i& yThreshold, const __m256i& yStrength, const __m256i& yWc, const __m256i& yBc) {
	__m256i y0, y1, y2, ySrc0, ySrc1, ySrc2, yTemp;
	__m256i yY, yVmin, yVmax, yMin, yMax, yMask, yAvg;
	ySrc0 = get_previous_2_y_pixels(src);
	ySrc1 = get_y_from_pixelyc(src);
	const BYTE *src_fin = src + w * PIXELYC_SIZE;
	for ( ; src < src_fin; src += 96, dst += 96) {
		//周辺近傍の最大と最小を縦方向・横方向に求める
		yVmax = get_y_from_pixelyc(src + (-2*max_w) * PIXELYC_SIZE);
		yVmin = yVmax;
		yTemp = get_y_from_pixelyc(src + (-1*max_w) * PIXELYC_SIZE);
		yVmax = _mm256_max_epi16(yVmax, yTemp);
		yVmin = _mm256_min_epi16(yVmin, yTemp);
		yVmax = _mm256_max_epi16(yVmax, ySrc1);
		yVmin = _mm256_min_epi16(yVmin, ySrc1);
		yTemp = get_y_from_pixelyc(src + (1*max_w) * PIXELYC_SIZE);
		yVmax = _mm256_max_epi16(yVmax, yTemp);
		yVmin = _mm256_min_epi16(yVmin, yTemp);
		yTemp = get_y_from_pixelyc(src + (2*max_w) * PIXELYC_SIZE);
		yVmax = _mm256_max_epi16(yVmax, yTemp);
		yVmin = _mm256_min_epi16(yVmin, yTemp);
		ySrc2 = get_y_from_pixelyc(src + 16 * PIXELYC_SIZE);
		yMax  = ySrc1;
		yMin  = yMax;
		yTemp = _mm256_alignr256_epi8(ySrc1, ySrc0, 32-4);
		yMax  = _mm256_max_epi16(yMax, yTemp);
		yMin  = _mm256_min_epi16(yMin, yTemp);
		yTemp = _mm256_alignr256_epi8(ySrc1, ySrc0, 32-2);
		yMax  = _mm256_max_epi16(yMax, yTemp);
		yMin  = _mm256_min_epi16(yMin, yTemp);
		yTemp = _mm256_alignr256_epi8(ySrc2, ySrc1, 2);
		yMax  = _mm256_max_epi16(yMax, yTemp);
		yMin  = _mm256_min_epi16(yMin, yTemp);
		yTemp = _mm256_alignr256_epi8(ySrc2, ySrc1, 4);
		yMax  = _mm256_max_epi16(yMax, yTemp);
		yMin  = _mm256_min_epi16(yMin, yTemp);
		ySrc0 = ySrc1;
		ySrc1 = ySrc2;
				
		//if (max - min < vmax - vmin) { max = vmax, min = vmin; }
		yMask = _mm256_cmpgt_epi16(_mm256_sub_epi16(yVmax, yVmin), _mm256_sub_epi16(yMax, yMin));
		yMax  = _mm256_blendv_epi8(yMax, yVmax, yMask);
		yMin  = _mm256_blendv_epi8(yMin, yVmin, yMask);
				
		//avg = (min + max) >> 1;
		yAvg  = _mm256_add_epi16(yMax, yMin);
		yAvg  = _mm256_srai_epi16(yAvg, 1);

		//if (max - min > thrs)
		yMask = _mm256_cmpgt_epi16(_mm256_sub_epi16(yMax, yMin), yThreshold);

		//if (src->y == max) max += wc * 2;
		//else max += wc;
		y1    = _mm256_cmpeq_epi16(ySrc0, yMax);
		yMax  = _mm256_add_epi16(yMax, yWc);
		yMax  = _mm256_add_epi16(yMax, _mm256_and_si256(yWc, y1));
				
		//if (src->y == min) min -= bc * 2;
		//else  min -= bc;
		y1    = _mm256_cmpeq_epi16(ySrc0, yMin);
		yMin  = _mm256_sub_epi16(yMin, yBc);
		yMin  = _mm256_sub_epi16(yMin, _mm256_and_si256(yBc, y1));

		//dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
		y1    = _mm256_sub_epi16(yAvg, ySrc0);
		y0    = _mm256_unpacklo_epi16(y1, y1);
		y1    = _mm256_unpackhi_epi16(y1, y1);
		y0    = _mm256_madd_epi16(y0, yStrength);
		y1    = _mm256_madd_epi16(y1, yStrength);
		y0    = _mm256_srai_epi32(y0, 4);
		y1    = _mm256_srai_epi32(y1, 4);
		y0    = _mm256_packs_epi32(y0, y1);
		y0    = _mm256_add_epi16(y0, ySrc0);
		y0    = _mm256_max_epi16(y0, yMin);
		y0    = _mm256_min_epi16(y0, yMax);

		yY    = _mm256_blendv_epi8(ySrc0, y0, yMask);

		y0 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src + 48)), _mm_loadu_si128((__m128i *)(src +  0))); // 384,   0
		y1 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src + 64)), _mm_loadu_si128((__m128i *)(src + 16))); // 512, 128
		y2 = _mm256_set_m128i(_mm_loadu_si128((__m128i *)(src + 80)), _mm_loadu_si128((__m128i *)(src + 32))); // 768, 256

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
}

void multi_thread_func_avx2(int thread_id, int thread_num, void *param1, void *param2) {
//	thread_id	: スレッド番号 ( 0 ～ thread_num-1 )
//	thread_num	: スレッド数 ( 1 ～ )
//	param1		: 汎用パラメータ
//	param2		: 汎用パラメータ
//
//	この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
	FILTER *fp				= (FILTER *)param1;
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param2;

	const int max_w = fpip->max_w;
	const int h = fpip->h, w = fpip->w;
	const int str = fp->track[0], thrs = fp->track[1] << 3;
	const int bc = fp->track[2] << 4, wc = fp->track[3] << 4;
	const __m256i yStrength = _mm256_unpacklo_epi16(_mm256_setzero_si256(), _mm256_set1_epi16((short)(-1 * str)));
	const __m256i yThreshold = _mm256_set1_epi16((short)thrs);
	const __m256i yBc = _mm256_set1_epi16((short)bc);
	const __m256i yWc = _mm256_set1_epi16((short)wc);

	//	スレッド毎の画像を処理する場所を計算する
	const int y_start = (h *  thread_id   ) / thread_num;
	const int y_end   = (h * (thread_id+1)) / thread_num;

	BYTE *line_src = (BYTE *)fpip->ycp_edit + y_start * max_w * PIXELYC_SIZE;
	BYTE *line_dst = (BYTE *)fpip->ycp_temp + y_start * max_w * PIXELYC_SIZE;
	for (int y = y_start; y < y_end; y++, line_src += max_w * PIXELYC_SIZE, line_dst += max_w * PIXELYC_SIZE) {
		if (1 < y && y < h - 2) {
			int process_pixel = w;
			BYTE *src = line_src;
			BYTE *dst = line_dst;
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
