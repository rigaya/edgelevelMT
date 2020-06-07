//---------------------------------------------------------------------------------------------
//      マルチスレッド対応サンプルフィルタ(フィルタプラグイン)  for AviUtl ver0.99a以降
//---------------------------------------------------------------------------------------------
#define _CRT_SECURE_NO_WARNINGS
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <tchar.h>
#include <algorithm>
#include "filter.h"
#include "edgelevelMT.h"
#include "rgy_perf_checker.h"

auto perfCheck = RGYPerfChecker<ENABLE_PERF_CHECK, 1>({ "edgelevel" });
enum {
    EDGELEVEL_FUNC,
};

//---------------------------------------------------------------------
//      フィルタ構造体定義
//---------------------------------------------------------------------
const int TRACK_N = 4;
TCHAR *track_name[]    = { (TCHAR*)"特性", (TCHAR*)"閾値", (TCHAR*)"黒補正", (TCHAR*)"白補正" };
int    track_default[] = { 10, 16, 0, 0 };
int    track_s[]       = { -31, 0, 0, 0 };
int    track_e[]       = { 31, 255, 31, 31 };

const int CHECK_N = 1;
TCHAR *check_name[]    = { (TCHAR*)"判定表示" };
int    check_default[] = { 0 };

FILTER_DLL filter = {
    FILTER_FLAG_EX_INFORMATION, //  フィルタのフラグ
                                //  FILTER_FLAG_ALWAYS_ACTIVE       : フィルタを常にアクティブにします
                                //  FILTER_FLAG_CONFIG_POPUP        : 設定をポップアップメニューにします
                                //  FILTER_FLAG_CONFIG_CHECK        : 設定をチェックボックスメニューにします
                                //  FILTER_FLAG_CONFIG_RADIO        : 設定をラジオボタンメニューにします
                                //  FILTER_FLAG_EX_DATA             : 拡張データを保存出来るようにします。
                                //  FILTER_FLAG_PRIORITY_HIGHEST    : フィルタのプライオリティを常に最上位にします
                                //  FILTER_FLAG_PRIORITY_LOWEST     : フィルタのプライオリティを常に最下位にします
                                //  FILTER_FLAG_WINDOW_THICKFRAME   : サイズ変更可能なウィンドウを作ります
                                //  FILTER_FLAG_WINDOW_SIZE         : 設定ウィンドウのサイズを指定出来るようにします
                                //  FILTER_FLAG_DISP_FILTER         : 表示フィルタにします
                                //  FILTER_FLAG_EX_INFORMATION      : フィルタの拡張情報を設定できるようにします
                                //  FILTER_FLAG_NO_CONFIG           : 設定ウィンドウを表示しないようにします
                                //  FILTER_FLAG_AUDIO_FILTER        : オーディオフィルタにします
                                //  FILTER_FLAG_RADIO_BUTTON        : チェックボックスをラジオボタンにします
                                //  FILTER_FLAG_WINDOW_HSCROLL      : 水平スクロールバーを持つウィンドウを作ります
                                //  FILTER_FLAG_WINDOW_VSCROLL      : 垂直スクロールバーを持つウィンドウを作ります
                                //  FILTER_FLAG_IMPORT              : インポートメニューを作ります
                                //  FILTER_FLAG_EXPORT              : エクスポートメニューを作ります
    0,0,                        //  設定ウインドウのサイズ (FILTER_FLAG_WINDOW_SIZEが立っている時に有効)
    AUF_FULL_NAME,              //  フィルタの名前
    TRACK_N,                    //  トラックバーの数 (0なら名前初期値等もNULLでよい)
    track_name,                 //  トラックバーの名前郡へのポインタ
    track_default,              //  トラックバーの初期値郡へのポインタ
    track_s,track_e,            //  トラックバーの数値の下限上限 (NULLなら全て0～256)
    CHECK_N,                    //  チェックボックスの数 (0なら名前初期値等もNULLでよい)
    check_name,                 //  チェックボックスの名前郡へのポインタ
    check_default,              //  チェックボックスの初期値郡へのポインタ
    func_proc,                  //  フィルタ処理関数へのポインタ (NULLなら呼ばれません)
    func_init,                  //  開始時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                       //  終了時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                       //  設定が変更されたときに呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                       //  設定ウィンドウにウィンドウメッセージが来た時に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,NULL,                  //  システムで使いますので使用しないでください
    NULL,                       //  拡張データ領域へのポインタ (FILTER_FLAG_EX_DATAが立っている時に有効)
    NULL,                       //  拡張データサイズ (FILTER_FLAG_EX_DATAが立っている時に有効)
    AUF_VERSION_NAME,
                                //  フィルタ情報へのポインタ (FILTER_FLAG_EX_INFORMATIONが立っている時に有効)
    NULL,                       //  セーブが開始される直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
    NULL,                       //  セーブが終了した直前に呼ばれる関数へのポインタ (NULLなら呼ばれません)
};

//---------------------------------------------------------------------
//      グローバル変数
//---------------------------------------------------------------------
// mt_func[0]    ... 通常処理用関数配列 (SIMD検出後設定)
// mt_func[1]    ... エッジ判定チェック用関数配列
// mt_func[n][0] ... non-aligned用関数
// mt_func[n][1] ... aligned用関数
static const MULTI_THREAD_FUNC *mt_func = NULL;

//---------------------------------------------------------------------
//      フィルタ構造体のポインタを渡す関数
//---------------------------------------------------------------------
EXTERN_C FILTER_DLL __declspec(dllexport) * __stdcall GetFilterTable(void)
{
    return &filter;
}

#pragma warning (push)
#pragma warning (disable: 4100) //warning C4100: 引数は関数の本体部で 1 度も参照されません。
BOOL func_init(FILTER *fp) {
    mt_func = get_func_list();
    return TRUE;
}
#pragma warning (pop)

//---------------------------------------------------------------------
//      フィルタ処理関数
//---------------------------------------------------------------------

//エッジレベル調整の基本コード
void multi_thread_func(int thread_id, int thread_num, void *param1, void *param2)
{
//  thread_id   : スレッド番号 ( 0 ～ thread_num-1 )
//  thread_num  : スレッド数 ( 1 ～ )
//  param1      : 汎用パラメータ
//  param2      : 汎用パラメータ
//
//  この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
//
    FILTER *fp              = (FILTER *)param1;
    FILTER_PROC_INFO *fpip  = (FILTER_PROC_INFO *)param2;

    PIXEL_YC *src, *dst;
    int pitch = fpip->max_w;
    const int h = fpip->h, w = fpip->w;
    const short str = (short)fp->track[0], thrs = (short)fp->track[1] * 8;
    const short bc = (short)fp->track[2] * 16, wc = (short)fp->track[3] * 16;
    short max, min, vmax, vmin, avg;

    //  スレッド毎の画像を処理する場所を計算する
    const int y_start = (h *  thread_id   ) / thread_num;
    const int y_end   = (h * (thread_id+1)) / thread_num;

    for (int y = y_start; y < y_end; y++) {
        src = fpip->ycp_edit + y * pitch;
        dst = fpip->ycp_temp + y * pitch;

        if (1 < y && y < h - 2) {
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
                    //if (src->y == min)
                    //  min -= bc * 2;
                    //else
                    //  min -= bc;
                    //if (src->y == max)
                    //  max += wc * 2;
                    //else
                    //  max += wc;

                    if (src->y == min)
                        min -= bc;
                    min -= bc;
                    if (src->y == max)
                        max += wc;
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
        } else {
            for (int x = 0; x < w; x++)
                dst[x] = src[x];
        }
    }
}

//エッジと判定して、調整をかけるところを表示する
//調整をかけないところは黒
void multi_thread_check_threshold(int thread_id, int thread_num, void *param1, void *param2)
{
    //  thread_id   : スレッド番号 ( 0 ～ thread_num-1 )
    //  thread_num  : スレッド数 ( 1 ～ )
    //  param1      : 汎用パラメータ
    //  param2      : 汎用パラメータ
    //
    //  この関数内からWin32APIや外部関数(rgb2yc,yc2rgbは除く)を使用しないでください。
    //
    FILTER *fp              = (FILTER *)param1;
    FILTER_PROC_INFO *fpip  = (FILTER_PROC_INFO *)param2;

    PIXEL_YC *src, *dst;
    const int pitch = fpip->max_w;
    const int h = fpip->h, w = fpip->w;
    const int thrs = fp->track[1] * 8;
    short max, min, vmax, vmin, avg;

    //  スレッド毎の画像を処理する場所を計算する
    const int y_start = (h *  thread_id   ) / thread_num;
    const int y_end   = (h * (thread_id+1)) / thread_num;

    const PIXEL_YC YC_ORANGE = { 2255, -836,  1176 }; //調整 - 明 - 白補正対象
    const PIXEL_YC YC_YELLOW = { 3514, -626,    73 }; //調整 - 明
    const PIXEL_YC YC_SKY    = { 3702,  169,  -610 }; //調整 - 暗
    const PIXEL_YC YC_BLUE   = { 1900, 1240,  -230 }; //調整 - 暗 - 黒補正対象
    const PIXEL_YC YC_BLACK  = { 1013,    0,     0 }; //エッジでない

    for (int y = y_start; y < y_end; y++) {
        src = fpip->ycp_edit + y * pitch;
        dst = fpip->ycp_temp + y * pitch;

        if (1 < y && y < h - 2) {
            *dst = YC_BLACK;
            dst++; src++;
            *dst = YC_BLACK;
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
                    avg = (max + min) >> 1;
                    if (src->y == min)
                        *dst = YC_BLUE;
                    else if (src->y == max)
                        *dst = YC_ORANGE;
                    else
                        *dst = (src->y > avg) ? YC_YELLOW : YC_SKY;
                } else {
                    *dst = YC_BLACK;
                }

                dst++; src++;
            }
            for (int x = n; x < w; x++) {
                *dst = YC_BLACK;
                dst++; src++;
            }
        } else {
            for (int x = 0; x < w; x++)
                dst[x] = YC_BLACK;
        }
    }
}

void simd_debug(FILTER *fp, FILTER_PROC_INFO *fpip) {
    if (fp->check[0])
        return;

    PIXEL_YC *debug_buffer = (PIXEL_YC *)_aligned_malloc(fpip->max_w * fpip->max_h * sizeof(PIXEL_YC), 16);
    if (NULL == debug_buffer)
        return;

    PIXEL_YC *temp = fpip->ycp_temp;
    fpip->ycp_temp = debug_buffer;
    fp->exfunc->exec_multi_thread_func(multi_thread_func, (void *)fp, (void *)fpip);

    int error_count = 0;
    for (int y = 0; y < fpip->h; y++) {
        PIXEL_YC *ptr_simd  = temp + y * fpip->max_w;
        PIXEL_YC *ptr_debug = debug_buffer + y * fpip->max_w;
        for (int x = 0; x < fpip->w; x++) {
            if (0 != memcmp(&ptr_simd[x], &ptr_debug[x], sizeof(PIXEL_YC))) {
                error_count++;
            }
        }
    }
    if (error_count)
        MessageBox(NULL, "SIMD error!", "edgelevelMT", NULL);

    fpip->ycp_temp = temp;
    _aligned_free(debug_buffer);
}

//---------------------------------------------------------------------
//      フィルタ処理関数
//---------------------------------------------------------------------
BOOL func_proc(FILTER *fp, FILTER_PROC_INFO *fpip)
{
    BOOL run = !fp->check[0] | fp->exfunc->is_saving(fpip->editp);
    BOOL align = (((size_t)fpip->ycp_temp | (size_t)fpip->ycp_edit | fpip->max_w) & 0x0F) == 0x00;
    perfCheck.settime(0);
    //  マルチスレッドでフィルタ処理関数を呼ぶ
    fp->exfunc->exec_multi_thread_func(
        mt_func[run + (align & run)],
        (void *)fp, (void *)fpip);
    perfCheck.settime(1);
    perfCheck.setcounter();
    perfCheck.print("edgeleveMT.csv", 1024);

#if SIMD_DEBUG
    simd_debug(fp, fpip);
#endif

    //  もし画像領域ポインタの入れ替えや解像度変更等の
    //  fpip の内容を変える場合はこちらの関数内で処理をする
    std::swap(fpip->ycp_edit, fpip->ycp_temp);

    return TRUE;
}
