#pragma once

#include <Windows.h>

#define AUF_VERSION      0.7
#define AUF_VERSION_STR  "0.7 v2"
#define AUF_NAME         "edgelevelMT.auf"
#define AUF_FULL_NAME    "エッジレベル調整 MT ver 0.7"
#define AUF_VERSION_NAME "エッジレベル調整 MT ver 0.7 v2"
#define AUF_VERSION_INFO AUF_VERSION_NAME

#ifdef DEBUG
#define VER_DEBUG   VS_FF_DEBUG
#define VER_PRIVATE VS_FF_PRIVATEBUILD
#else
#define VER_DEBUG   0
#define VER_PRIVATE 0
#endif

#define VER_STR_COMMENTS         AUF_FULL_NAME
#define VER_STR_COMPANYNAME      ""
#define VER_STR_FILEDESCRIPTION  AUF_FULL_NAME
#define VER_FILEVERSION          AUF_VERSION
#define VER_STR_FILEVERSION      AUF_VERSION_STR
#define VER_STR_INTERNALNAME     AUF_FULL_NAME
#define VER_STR_ORIGINALFILENAME AUF_NAME
#define VER_STR_LEGALCOPYRIGHT   AUF_FULL_NAME
#define VER_STR_PRODUCTNAME      "edgelevelMT"
#define VER_PRODUCTVERSION       VER_FILEVERSION
#define VER_STR_PRODUCTVERSION   VER_STR_FILEVERSION


#define PIXELYC_SIZE 6

#define ALIGN16_CONST_ARRAY static const _declspec(align(16))

ALIGN16_CONST_ARRAY BYTE   Array_SUFFLE_YCP_Y[]      = {0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11};
ALIGN16_CONST_ARRAY USHORT Array_MASK_YCP_SELECT_Y[] = {0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0xFFFF, 0x0000};
ALIGN16_CONST_ARRAY USHORT Array_MASK_FRAME_EDGE[]   = {0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF, 0x0000, 0x0000};

#define SUFFLE_YCP_Y       _mm_load_si128((__m128i*)Array_SUFFLE_YCP_Y)
#define MASK_YCP_SELECT_Y  _mm_load_si128((__m128i*)Array_MASK_YCP_SELECT_Y)
#define MASK_FRAME_EDGE    _mm_load_si128((__m128i*)Array_MASK_FRAME_EDGE)
