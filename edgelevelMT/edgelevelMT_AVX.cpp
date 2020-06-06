#include <Windows.h>
#include "filter.h"
#include "edgelevelMT.h"

//----------------------------------------------------------------------------
//    128bit-AVX 版 エッジレベル調整
//     SSE4.1 版を /arch:AVXでコンパイル
//----------------------------------------------------------------------------
#include <smmintrin.h> //for SSE4_1

static inline __m128i get_y_from_pixelyc_sse4_1(BYTE *src) {
	const int MASK_INT = 0x40 + 0x08 + 0x01;
	__m128i x0 = _mm_loadu_si128((__m128i *)(src +  0));
	__m128i x1 = _mm_loadu_si128((__m128i *)(src + 16));
	__m128i x2 = _mm_loadu_si128((__m128i *)(src + 32));
	x2 = _mm_blend_epi16(x2, x0, MASK_INT);
	x2 = _mm_blend_epi16(x2, x1, MASK_INT<<1);
	x2 = _mm_shuffle_epi8(x2, SUFFLE_YCP_Y);
	return x2;
}

static inline __m128i get_y_from_pixelyc_sse4_1_aligned(BYTE *src) {
	const int MASK_INT = 0x40 + 0x08 + 0x01;
	__m128i x0 = _mm_load_si128((__m128i *)(src +  0));
	__m128i x1 = _mm_load_si128((__m128i *)(src + 16));
	__m128i x2 = _mm_load_si128((__m128i *)(src + 32));
	x2 = _mm_blend_epi16(x2, x0, MASK_INT);
	x2 = _mm_blend_epi16(x2, x1, MASK_INT<<1);
	x2 = _mm_shuffle_epi8(x2, SUFFLE_YCP_Y);
	return x2;
}


void multi_thread_func_avx(int thread_id, int thread_num, void *param1, void *param2) {
//	thread_id	: スレッド番号 ( 0 ～ thread_num-1 )
//	thread_num	: スレッド数 ( 1 ～ )
//	param1		: 汎用パラメータ
//	param2		: 汎用パラメータ
//
//	この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
	FILTER *fp				= (FILTER *)param1;
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param2;

	BYTE * const ycp_edt = (BYTE *)fpip->ycp_edit;
	BYTE * const ycp_tmp = (BYTE *)fpip->ycp_temp;
	BYTE *src, *dst, *src_fin, *line_src, *line_dst;
	const int max_w = fpip->max_w;
	const int h = fpip->h, w = fpip->w;
	const int str = fp->track[0], thrs = fp->track[1] * 8;
	const int bc = fp->track[2] * 16, wc = fp->track[3] * 16;
	//こんなにレジスタは多くないけど、適当に最適化してもらう
	__m128i x0, x1, x2, x3;
	__m128i xY, xVmin, xVmax, xMin, xMax, xMask, xAvg;
	const int MASK_INT = 0x40 + 0x08 + 0x01;

	//	スレッド毎の画像を処理する場所を計算する
	const int y_start = (h *  thread_id   ) / thread_num;
	const int y_end   = (h * (thread_id+1)) / thread_num;

	for (int y = y_start; y < y_end; y++) {
		src = ycp_edt + y * max_w * PIXELYC_SIZE;
		dst = ycp_tmp + y * max_w * PIXELYC_SIZE;
		src_fin = src + w * PIXELYC_SIZE;
		if (1 < y && y < h - 2) {
			line_src = src;
			line_dst = dst;
			for ( ; src < src_fin; src += 48, dst += 48) {
				//周辺近傍の最大と最小を縦方向・横方向に求める
				xVmax = get_y_from_pixelyc_sse4_1(src + (-2*max_w) * PIXELYC_SIZE);
				xVmin = xVmax;
				x0    = get_y_from_pixelyc_sse4_1(src + (-1*max_w) * PIXELYC_SIZE);
				xVmax = _mm_max_epi16(xVmax, x0);
				xVmin = _mm_min_epi16(xVmin, x0);
				x0    = get_y_from_pixelyc_sse4_1(src + (0 - 2) * PIXELYC_SIZE);
				x1    = get_y_from_pixelyc_sse4_1(src + (8 - 2) * PIXELYC_SIZE);
				xMax  = x0;
				xMin  = xMax;
				x2    = _mm_alignr_epi8(x1, x0, 2);
				xMax  = _mm_max_epi16(xMax, x2);
				xMin  = _mm_min_epi16(xMin, x2);
				xY    = _mm_alignr_epi8(x1, x0, 4);
				xMax  = _mm_max_epi16(xMax, xY);
				xMin  = _mm_min_epi16(xMin, xY);
				xVmax = _mm_max_epi16(xVmax, xY);
				xVmin = _mm_min_epi16(xVmin, xY);
				x2    = _mm_alignr_epi8(x1, x0, 6);
				xMax  = _mm_max_epi16(xMax, x2);
				xMin  = _mm_min_epi16(xMin, x2);
				x2    = _mm_alignr_epi8(x1, x0, 8);
				xMax  = _mm_max_epi16(xMax, x2);
				xMin  = _mm_min_epi16(xMin, x2);
				x0    = get_y_from_pixelyc_sse4_1(src + (1*max_w) * PIXELYC_SIZE);
				xVmax = _mm_max_epi16(xVmax, x0);
				xVmin = _mm_min_epi16(xVmin, x0);
				x0    = get_y_from_pixelyc_sse4_1(src + (2*max_w) * PIXELYC_SIZE);
				xVmax = _mm_max_epi16(xVmax, x0);
				xVmin = _mm_min_epi16(xVmin, x0);
				
				//if (max - min < vmax - vmin) { max = vmax, min = vmin; }
				xMask = _mm_cmpgt_epi16(_mm_sub_epi16(xVmax, xVmin), _mm_sub_epi16(xMax, xMin));
				xMax  = _mm_blendv_epi8(xMax, xVmax, xMask);
				xMin  = _mm_blendv_epi8(xMin, xVmin, xMask);
				
				//avg = (min + max) >> 1;
				xAvg  = _mm_add_epi16(xMax, xMin);
				xAvg  = _mm_srai_epi16(xAvg, 1);

				//if (max - min > thrs)
				xMask = _mm_cmpgt_epi16(_mm_sub_epi16(xMax, xMin), _mm_set1_epi16(thrs));

				//if (src->y == max) max += wc * 2;
				//else max += wc;
				x1    = _mm_cmpeq_epi16(xY, xMax);
				xMax  = _mm_add_epi16(xMax, _mm_set1_epi16(wc));
				xMax  = _mm_add_epi16(xMax, _mm_and_si128(_mm_set1_epi16(wc), x1));
				
				//if (src->y == min) min -= bc * 2;
				//else  min -= bc;
				x1    = _mm_cmpeq_epi16(xY, xMin);
				xMin  = _mm_sub_epi16(xMin, _mm_set1_epi16(bc));
				xMin  = _mm_sub_epi16(xMin, _mm_and_si128(_mm_set1_epi16(bc), x1));

				//dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
				x1    = _mm_sub_epi16(xAvg, xY);
				x0    = _mm_unpacklo_epi16(x1, x1);
				x1    = _mm_unpackhi_epi16(x1, x1);
				x2    = _mm_unpacklo_epi16(_mm_setzero_si128(), _mm_set1_epi16(-1 * str));
				x0    = _mm_madd_epi16(x0, x2);
				x1    = _mm_madd_epi16(x1, x2);
				x0    = _mm_srai_epi32(x0, 4);
				x1    = _mm_srai_epi32(x1, 4);
				x0    = _mm_packs_epi32(x0, x1);
				x0    = _mm_add_epi16(x0, xY);
				x0    = _mm_max_epi16(x0, xMin);
				x0    = _mm_min_epi16(x0, xMax);

				xY    = _mm_blendv_epi8(xY, x0, xMask);

				x0 = _mm_loadu_si128((__m128i *)(src +  0));
				x1 = _mm_loadu_si128((__m128i *)(src + 16));
				x2 = _mm_loadu_si128((__m128i *)(src + 32));
				
				xY = _mm_shuffle_epi8(xY, SUFFLE_YCP_Y);
				x0 = _mm_blend_epi16(x0, xY, MASK_INT);
				x1 = _mm_blend_epi16(x1, xY, MASK_INT<<1);
				x2 = _mm_blend_epi16(x2, xY, (MASK_INT<<2) & 0xFF);
				
				_mm_storeu_si128((__m128i *)(dst +  0), x0);
				_mm_storeu_si128((__m128i *)(dst + 16), x1);
				_mm_storeu_si128((__m128i *)(dst + 32), x2);
			}
			x0 = _mm_loadu_si128((__m128i *)line_src);
			x1 = _mm_loadu_si128((__m128i *)(line_src + w * PIXELYC_SIZE - 12));
			_mm_maskmoveu_si128(x0, MASK_FRAME_EDGE, (char *)line_dst);
			_mm_storeu_si128((__m128i *)(line_dst + w * PIXELYC_SIZE - 12), x1);
		} else {
			src_fin = (BYTE *)((size_t)src_fin & (~63));
			for ( ; src < src_fin; src += 64, dst += 64) {
				x0 = _mm_loadu_si128((__m128i *)(src +  0));
				x1 = _mm_loadu_si128((__m128i *)(src + 16));
				x2 = _mm_loadu_si128((__m128i *)(src + 32));
				x3 = _mm_loadu_si128((__m128i *)(src + 48));
				_mm_storeu_si128((__m128i *)(dst +  0), x0);
				_mm_storeu_si128((__m128i *)(dst + 16), x1);
				_mm_storeu_si128((__m128i *)(dst + 32), x2);
				_mm_storeu_si128((__m128i *)(dst + 48), x3);
			}
			if ((w * PIXELYC_SIZE) & 32) {
				x0 = _mm_loadu_si128((__m128i *)(src +  0));
				x1 = _mm_loadu_si128((__m128i *)(src + 16));
				_mm_storeu_si128((__m128i *)(dst +  0), x0);
				_mm_storeu_si128((__m128i *)(dst + 16), x1);
				src += 32, dst += 32;
			}
			if ((w * PIXELYC_SIZE) & 16) {
				x0 = _mm_loadu_si128((__m128i *)src);
				_mm_storeu_si128((__m128i *)dst, x0);
				src += 16, dst += 16;
			}
			src_fin = src + ((w * PIXELYC_SIZE) & 15);
			for ( ; src < src_fin; src += 2, dst += 2)
				*(short *)dst = *(short *)src;
		}
	}
}

void multi_thread_func_avx_aligned(int thread_id, int thread_num, void *param1, void *param2) {
//	thread_id	: スレッド番号 ( 0 ～ thread_num-1 )
//	thread_num	: スレッド数 ( 1 ～ )
//	param1		: 汎用パラメータ
//	param2		: 汎用パラメータ
//
//	この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
	FILTER *fp				= (FILTER *)param1;
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param2;

	BYTE * const ycp_edt = (BYTE *)fpip->ycp_edit;
	BYTE * const ycp_tmp = (BYTE *)fpip->ycp_temp;
	BYTE *src, *dst, *src_fin, *line_src, *line_dst;
	const int max_w = fpip->max_w;
	const int h = fpip->h, w = fpip->w;
	const int str = fp->track[0], thrs = fp->track[1] * 8;
	const int bc = fp->track[2] * 16, wc = fp->track[3] * 16;
	//こんなにレジスタは多くないけど、適当に最適化してもらう
	__m128i x0, x1, x2, x3;
	__m128i xY, xVmin, xVmax, xMin, xMax, xMask, xAvg;
	const int MASK_INT = 0x40 + 0x08 + 0x01;

	//	スレッド毎の画像を処理する場所を計算する
	const int y_start = (h *  thread_id   ) / thread_num;
	const int y_end   = (h * (thread_id+1)) / thread_num;

	for (int y = y_start; y < y_end; y++) {
		src = ycp_edt + y * max_w * PIXELYC_SIZE;
		dst = ycp_tmp + y * max_w * PIXELYC_SIZE;
		src_fin = src + w * PIXELYC_SIZE;
		if (1 < y && y < h - 2) {
			line_src = src;
			line_dst = dst;
			for ( ; src < src_fin; src += 48, dst += 48) {
				//周辺近傍の最大と最小を縦方向・横方向に求める
				xVmax = get_y_from_pixelyc_sse4_1_aligned(src + (-2*max_w) * PIXELYC_SIZE);
				xVmin = xVmax;
				x0    = get_y_from_pixelyc_sse4_1_aligned(src + (-1*max_w) * PIXELYC_SIZE);
				xVmax = _mm_max_epi16(xVmax, x0);
				xVmin = _mm_min_epi16(xVmin, x0);
				x0    = get_y_from_pixelyc_sse4_1(src + (0 - 2) * PIXELYC_SIZE);
				x1    = get_y_from_pixelyc_sse4_1(src + (8 - 2) * PIXELYC_SIZE);
				xMax  = x0;
				xMin  = x0;
				x2    = _mm_alignr_epi8(x1, x0, 2);
				xMax  = _mm_max_epi16(xMax, x2);
				xMin  = _mm_min_epi16(xMin, x2);
				xY    = _mm_alignr_epi8(x1, x0, 4);
				xMax  = _mm_max_epi16(xMax, xY);
				xMin  = _mm_min_epi16(xMin, xY);
				xVmax = _mm_max_epi16(xVmax, xY);
				xVmin = _mm_min_epi16(xVmin, xY);
				x2    = _mm_alignr_epi8(x1, x0, 6);
				xMax  = _mm_max_epi16(xMax, x2);
				xMin  = _mm_min_epi16(xMin, x2);
				x2    = _mm_alignr_epi8(x1, x0, 8);
				xMax  = _mm_max_epi16(xMax, x2);
				xMin  = _mm_min_epi16(xMin, x2);
				x0    = get_y_from_pixelyc_sse4_1_aligned(src + (1*max_w) * PIXELYC_SIZE);
				xVmax = _mm_max_epi16(xVmax, x0);
				xVmin = _mm_min_epi16(xVmin, x0);
				x0    = get_y_from_pixelyc_sse4_1_aligned(src + (2*max_w) * PIXELYC_SIZE);
				xVmax = _mm_max_epi16(xVmax, x0);
				xVmin = _mm_min_epi16(xVmin, x0);
				
				//if (max - min < vmax - vmin) { max = vmax, min = vmin; }
				xMask = _mm_cmpgt_epi16(_mm_sub_epi16(xVmax, xVmin), _mm_sub_epi16(xMax, xMin));
				xMax  = _mm_blendv_epi8(xMax, xVmax, xMask);
				xMin  = _mm_blendv_epi8(xMin, xVmin, xMask);
				
				//avg = (min + max) >> 1;
				xAvg  = _mm_add_epi16(xMax, xMin);
				xAvg  = _mm_srai_epi16(xAvg, 1);

				//if (max - min > thrs)
				xMask = _mm_cmpgt_epi16(_mm_sub_epi16(xMax, xMin), _mm_set1_epi16(thrs));

				//if (src->y == max) max += wc * 2;
				//else max += wc;
				x1    = _mm_cmpeq_epi16(xY, xMax);
				xMax  = _mm_add_epi16(xMax, _mm_set1_epi16(wc));
				xMax  = _mm_add_epi16(xMax, _mm_and_si128(_mm_set1_epi16(wc), x1));
				
				//if (src->y == min) min -= bc * 2;
				//else  min -= bc;
				x1    = _mm_cmpeq_epi16(xY, xMin);
				xMin  = _mm_sub_epi16(xMin, _mm_set1_epi16(bc));
				xMin  = _mm_sub_epi16(xMin, _mm_and_si128(_mm_set1_epi16(bc), x1));

				//dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
				x1    = _mm_sub_epi16(xAvg, xY);
				x0    = _mm_unpacklo_epi16(x1, x1);
				x1    = _mm_unpackhi_epi16(x1, x1);
				x2    = _mm_unpacklo_epi16(_mm_setzero_si128(), _mm_set1_epi16(-1 * str));
				x0    = _mm_madd_epi16(x0, x2);
				x1    = _mm_madd_epi16(x1, x2);
				x0    = _mm_srai_epi32(x0, 4);
				x1    = _mm_srai_epi32(x1, 4);
				x0    = _mm_packs_epi32(x0, x1);
				x0    = _mm_add_epi16(x0, xY);
				x0    = _mm_max_epi16(x0, xMin);
				x0    = _mm_min_epi16(x0, xMax);

				xY    = _mm_blendv_epi8(xY, x0, xMask);

				x0 = _mm_load_si128((__m128i *)(src +  0));
				x1 = _mm_load_si128((__m128i *)(src + 16));
				x2 = _mm_load_si128((__m128i *)(src + 32));

				xY = _mm_shuffle_epi8(xY, SUFFLE_YCP_Y);
				x0 = _mm_blend_epi16(x0, xY, MASK_INT);
				x1 = _mm_blend_epi16(x1, xY, MASK_INT<<1);
				x2 = _mm_blend_epi16(x2, xY, (MASK_INT<<2) & 0xFF);
				
				_mm_stream_si128((__m128i *)(dst +  0), x0);
				_mm_stream_si128((__m128i *)(dst + 16), x1);
				_mm_stream_si128((__m128i *)(dst + 32), x2);
			}
			x0 = _mm_load_si128((__m128i *)line_src);
			x1 = _mm_loadu_si128((__m128i *)(line_src + w * PIXELYC_SIZE - 12));
			_mm_maskmoveu_si128(x0, MASK_FRAME_EDGE, (char *)line_dst);
			_mm_storeu_si128((__m128i *)(line_dst + w * PIXELYC_SIZE - 12), x1);
		} else {
			src_fin = (BYTE *)((size_t)src_fin & (~63));
			for ( ; src < src_fin; src += 64, dst += 64) {
				x0 = _mm_load_si128((__m128i *)(src +  0));
				x1 = _mm_load_si128((__m128i *)(src + 16));
				x2 = _mm_load_si128((__m128i *)(src + 32));
				x3 = _mm_load_si128((__m128i *)(src + 48));
				_mm_stream_si128((__m128i *)(dst +  0), x0);
				_mm_stream_si128((__m128i *)(dst + 16), x1);
				_mm_stream_si128((__m128i *)(dst + 32), x2);
				_mm_stream_si128((__m128i *)(dst + 48), x3);
			}
			if ((w * PIXELYC_SIZE) & 32) {
				x0 = _mm_load_si128((__m128i *)(src +  0));
				x1 = _mm_load_si128((__m128i *)(src + 16));
				_mm_stream_si128((__m128i *)(dst +  0), x0);
				_mm_stream_si128((__m128i *)(dst + 16), x1);
				src += 32, dst += 32;
			}
			if ((w * PIXELYC_SIZE) & 16) {
				x0 = _mm_load_si128((__m128i *)src);
				_mm_stream_si128((__m128i *)dst, x0);
				src += 16, dst += 16;
			}
			src_fin = src + ((w * PIXELYC_SIZE) & 15);
			for ( ; src < src_fin; src += 2, dst += 2)
				*(short *)dst = *(short *)src;
		}
	}
}
