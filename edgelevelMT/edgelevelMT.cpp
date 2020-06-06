//---------------------------------------------------------------------------------------------
//		マルチスレッド対応サンプルフィルタ(フィルタプラグイン)  for AviUtl ver0.99a以降
//---------------------------------------------------------------------------------------------
#include <Windows.h>
#include <algorithm>
#include "filter.h"
#include "edgelevelMT.h"
#include "edgelevelMT_util.h"


//---------------------------------------------------------------------
//		フィルタ構造体定義
//---------------------------------------------------------------------
const int TRACK_N = 4;
TCHAR *track_name[] =	{ (TCHAR*)"特性", (TCHAR*)"閾値", (TCHAR*)"黒補正", (TCHAR*)"白補正" };
int track_default[] =	{ 10,	16,	0,	0  };
int track_s[] =		{ -31,	0,	0,	0  };
int track_e[] =		{ 31,	255,	31,	31 };

const int CHECK_N = 0;

FILTER_DLL filter = {
	FILTER_FLAG_EX_INFORMATION,	//	フィルタのフラグ
								//	FILTER_FLAG_ALWAYS_ACTIVE		: フィルタを常にアクティブにします
								//	FILTER_FLAG_CONFIG_POPUP		: 設定をポップアップメニューにします
								//	FILTER_FLAG_CONFIG_CHECK		: 設定をチェックボックスメニューにします
								//	FILTER_FLAG_CONFIG_RADIO		: 設定をラジオボタンメニューにします
								//	FILTER_FLAG_EX_DATA				: 拡張データを保存出来るようにします。
								//	FILTER_FLAG_PRIORITY_HIGHEST	: フィルタのプライオリティを常に最上位にします
								//	FILTER_FLAG_PRIORITY_LOWEST		: フィルタのプライオリティを常に最下位にします
								//	FILTER_FLAG_WINDOW_THICKFRAME	: サイズ変更可能なウィンドウを作ります
								//	FILTER_FLAG_WINDOW_SIZE			: 設定ウィンドウのサイズを指定出来るようにします
								//	FILTER_FLAG_DISP_FILTER			: 表示フィルタにします
								//	FILTER_FLAG_EX_INFORMATION		: フィルタの拡張情報を設定できるようにします
								//	FILTER_FLAG_NO_CONFIG			: 設定ウィンドウを表示しないようにします
								//	FILTER_FLAG_AUDIO_FILTER		: オーディオフィルタにします
								//	FILTER_FLAG_RADIO_BUTTON		: チェックボックスをラジオボタンにします
								//	FILTER_FLAG_WINDOW_HSCROLL		: 水平スクロールバーを持つウィンドウを作ります
								//	FILTER_FLAG_WINDOW_VSCROLL		: 垂直スクロールバーを持つウィンドウを作ります
								//	FILTER_FLAG_IMPORT				: インポートメニューを作ります
								//	FILTER_FLAG_EXPORT				: エクスポートメニューを作ります
	0,0,						//	設定ウインドウのサイズ (FILTER_FLAG_WINDOW_SIZEが立っている時に有効)
	AUF_FULL_NAME,				//	フィルタの名前
	TRACK_N,					//	トラックバーの数 (0なら名前初期値等もNULLでよい)
	track_name,					//	トラックバーの名前郡へのポインタ
	track_default,				//	トラックバーの初期値郡へのポインタ
	track_s,track_e,			//	トラックバーの数値の下限上限 (NULLなら全て0～256)
	CHECK_N,					//	チェックボックスの数 (0なら名前初期値等もNULLでよい)
	NULL,					    //	チェックボックスの名前郡へのポインタ
	NULL,				        //	チェックボックスの初期値郡へのポインタ
	func_proc,					//	フィルタ処理関数へのポインタ (NULLなら呼ばれません)
	func_init,					//	開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	NULL,						//	終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	NULL,						//	設定が変更されたときに呼ばれる関数へのポインタ (NULLなら呼ばれません)
	NULL,						//	設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	NULL,NULL,					//	システムで使いますので使用しないでください
	NULL,						//  拡張データ領域へのポインタ (FILTER_FLAG_EX_DATAが立っている時に有効)
	NULL,						//  拡張データサイズ (FILTER_FLAG_EX_DATAが立っている時に有効)
	AUF_VERSION_NAME,
								//  フィルタ情報へのポインタ (FILTER_FLAG_EX_INFORMATIONが立っている時に有効)
	NULL,						//	セーブが開始される直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
	NULL,						//	セーブが終了した直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
};

//---------------------------------------------------------------------
//		エッジレベル調整 関数リスト
//---------------------------------------------------------------------
void multi_thread_func( int thread_id, int thread_num, void *param1, void *param2 );
void multi_thread_func_sse2(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_sse2_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_ssse3(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_ssse3_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_sse4_1(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_sse4_1_aligned(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_avx(int thread_id, int thread_num, void *param1, void *param2);
void multi_thread_func_avx_aligned(int thread_id, int thread_num, void *param1, void *param2);

static const MULTI_THREAD_FUNC func_list[][2] = {
	{ multi_thread_func,        multi_thread_func },
	{ multi_thread_func_sse2,   multi_thread_func_sse2_aligned },
	{ multi_thread_func_ssse3,  multi_thread_func_ssse3_aligned },
	{ multi_thread_func_sse4_1, multi_thread_func_sse4_1_aligned },
#if (_MSC_VER >= 1600)
	{ multi_thread_func_avx,    multi_thread_func_avx_aligned },
#endif
};


//---------------------------------------------------------------------
//		グローバル変数
//---------------------------------------------------------------------
static const MULTI_THREAD_FUNC *mt_func = NULL;

//---------------------------------------------------------------------
//		フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable( void )
{
	return &filter;
}

BOOL func_init(FILTER *fp) {
	DWORD availSIMD = get_availableSIMD();
	if      (availSIMD & AUF_SIMD_AVX)   mt_func = func_list[4];
	else if (availSIMD & AUF_SIMD_SSE41) mt_func = func_list[3];
	else if (availSIMD & AUF_SIMD_SSSE3) mt_func = func_list[2];
	else if (availSIMD & AUF_SIMD_SSE2)  mt_func = func_list[1];
	else                                 mt_func = func_list[0];
	return TRUE;
}

//---------------------------------------------------------------------
//		フィルタ処理関数
//---------------------------------------------------------------------
void multi_thread_func( int thread_id, int thread_num, void *param1, void *param2 )
{
//	thread_id	: スレッド番号 ( 0 ～ thread_num-1 )
//	thread_num	: スレッド数 ( 1 ～ )
//	param1		: 汎用パラメータ
//	param2		: 汎用パラメータ
//
//	この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
	FILTER *fp				= (FILTER *)param1;
	FILTER_PROC_INFO *fpip	= (FILTER_PROC_INFO *)param2;

	int	y_start, y_end;

	PIXEL_YC *src, *dst;
	int pitch = fpip->max_w;
	int h = fpip->h, w = fpip->w;
	int str = fp->track[0], thrs = fp->track[1] * 8;
	int bc = fp->track[2] * 16, wc = fp->track[3] * 16;
	short max, min, vmax, vmin, avg;

	//	スレッド毎の画像を処理する場所を計算する
	y_start = ( h * thread_id     ) / thread_num;
	y_end   = ( h * (thread_id+1) ) / thread_num;

	for (int y = y_start; y < y_end; y++) {
		src = fpip->ycp_edit + y * pitch;
		dst = fpip->ycp_temp + y * pitch;

		if (y <= 2 || h - 2 <= y) {
			for (int x = 0; x < w; x++)
				dst[x] = src[x];
		} else {
			*dst = *src;
			dst++; src++;
			*dst = *src;
			dst++; src++;

			int n = w - 2;
			for (int x = 2; x < n; x++) {
				max = min = src[-2].y;
				vmax = vmin = src[-2*pitch].y;

				for (int i = -1; i < 3; ++i) {
					max  = (std::max)( max, src[i].y );
					min  = (std::min)( min, src[i].y );
					vmax = (std::max)( vmax, src[i*pitch].y );
					vmin = (std::min)( vmin, src[i*pitch].y );
				}

				if (max - min < vmax - vmin)
					max = vmax, min = vmin;

				if (max - min > thrs) {
					avg = (min + max) >> 1;
					if (src->y == min)
						min -= bc * 2;
					else 
						min -= bc;
					if (src->y == max)
						max += wc * 2;
					else
						max += wc;

					dst->y = (std::min)( (std::max)( short( src->y + ((src->y - avg) * str >> 4) ), min ), max );
				} else
					dst->y = src->y;

				dst->cb = src->cb;
				dst->cr = src->cr;
				++src, ++dst;
			}
			for (int x = n; x < w; x++) {
				*dst = *src;
				dst++; src++;
			}
		}
	}
}

//---------------------------------------------------------------------
//		フィルタ処理関数
//---------------------------------------------------------------------
BOOL func_proc(FILTER *fp, FILTER_PROC_INFO *fpip)
{
	//	マルチスレッドでフィルタ処理関数を呼ぶ
	fp->exfunc->exec_multi_thread_func(
		mt_func[(((size_t)fpip->ycp_temp | (size_t)fpip->ycp_edit | fpip->max_w) & 0x0F) == 0x00], 
		(void *)fp, (void *)fpip);

	//	もし画像領域ポインタの入れ替えや解像度変更等の
	//	fpip の内容を変える場合はこちらの関数内で処理をする
	std::swap(fpip->ycp_edit, fpip->ycp_temp);

	return TRUE;
}
