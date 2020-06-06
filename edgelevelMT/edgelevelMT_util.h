#pragma once

#include <Windows.h>
#include <intrin.h>
#include <immintrin.h>

//SIMD
enum {
	AUF_SIMD_NONE  = 0x0000,
	AUF_SIMD_SSE2  = 0x0001,
	AUF_SIMD_SSE3  = 0x0002,
	AUF_SIMD_SSSE3 = 0x0004,
	AUF_SIMD_SSE41 = 0x0008,
	AUF_SIMD_SSE42 = 0x0010,
#if (_MSC_VER >= 1600)
	AUF_SIMD_AVX   = 0x0020,
	AUF_SIMD_AVX2  = 0x0040 
#else
	AUF_SIMD_AVX   = 0x0000,
	AUF_SIMD_AVX2  = 0x0000 
#endif
};

typedef DWORD AUF_SIMD; 

static AUF_SIMD get_availableSIMD() {
	int CPUInfo[4];
	__cpuid(CPUInfo, 1);
	AUF_SIMD simd = AUF_SIMD_NONE;
	if  (CPUInfo[3] & 0x04000000)
		simd |= AUF_SIMD_SSE2;
	if  (CPUInfo[2] & 0x00000001)
		simd |= AUF_SIMD_SSE3;
	if  (CPUInfo[2] & 0x00000200)
		simd |= AUF_SIMD_SSSE3;
	if  (CPUInfo[2] & 0x00080000)
		simd |= AUF_SIMD_SSE41;
	if  (CPUInfo[2] & 0x00100000)
		simd |= AUF_SIMD_SSE42;
#if (_MSC_VER >= 1600)
	UINT64 XGETBV = 0;
	if ((CPUInfo[2] & 0x18000000) == 0x18000000) {
		XGETBV = _xgetbv(0);
		if ((XGETBV & 0x06) == 0x06)
			simd |= AUF_SIMD_AVX;
	}
	//__cpuid(CPUInfo, 7);
	//if ((simd & AUF_SIMD_AVX) && (CPUInfo[1] & 0x00000020))
	//	simd |= AUF_SIMD_AVX2;
#endif
	return simd;
}
