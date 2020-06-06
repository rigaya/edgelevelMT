#pragma once

#include <Windows.h>

#define SIMD_DEBUG 0

#define AUF_VERSION      0,0,7,8
#define AUF_VERSION_STR  "0.7 v8"
#define AUF_NAME         "edgelevelMT.auf"
#define AUF_FULL_NAME    "エッジレベル調整 MT ver 0.7"
#define AUF_VERSION_NAME "エッジレベル調整 MT ver 0.7 v8"
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

const MULTI_THREAD_FUNC *get_func_list();
