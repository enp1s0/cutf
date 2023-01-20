#ifndef __CUTF_MATH_CUH__
#define __CUTF_MATH_CUH__

#include <algorithm>
#include <cuda_fp16.h>
#include <cmath>
#include "macro.hpp"
#include "experimental/fp.hpp"

#define DEF_TEMPLATE_MATH_FUNC_1(func) \
template<class T>  CUTF_DEVICE_FUNC inline T func(const T a);

#ifdef __CUDA_ARCH__
#define SPEC_MATH_FUNC_1_h( func ) \
template<> CUTF_DEVICE_FUNC inline half func<half>(const half a){return h##func( a );}	
#define SPEC_MATH_FUNC_1_h2( func ) \
template<> CUTF_DEVICE_FUNC inline half2 func<half2>(const half2 a){return h2##func( a );}	
#define SPEC_MATH_FUNC_1_f( func ) \
template<> CUTF_DEVICE_FUNC inline float func<float>(const float a){return func##f( a );}	
#define SPEC_MATH_FUNC_1_d( func ) \
template<> CUTF_DEVICE_FUNC inline double func<double>(const double a){return func( a );}	
#else
// Prototype only
#define SPEC_MATH_FUNC_1_h( func ) \
template<> CUTF_DEVICE_FUNC inline half func<half>(const half a);
#define SPEC_MATH_FUNC_1_h2( func ) \
template<> CUTF_DEVICE_FUNC inline half2 func<half2>(const half2 a);
#define SPEC_MATH_FUNC_1_f( func ) \
template<> CUTF_DEVICE_FUNC inline float func<float>(const float a);
#define SPEC_MATH_FUNC_1_d( func ) \
template<> CUTF_DEVICE_FUNC inline double func<double>(const double a);
#endif

#define MATH_FUNC(func) \
	DEF_TEMPLATE_MATH_FUNC_1(func) \
	SPEC_MATH_FUNC_1_h(func) \
	SPEC_MATH_FUNC_1_h2(func) \
	SPEC_MATH_FUNC_1_f(func) \
	SPEC_MATH_FUNC_1_d(func) \


#ifdef __CUDA_ARCH__
CUTF_DEVICE_FUNC inline float rcpf(const float a){return __frcp_rn(a);}
CUTF_DEVICE_FUNC inline double rcp(const double a){return __drcp_rn(a);}
#endif

namespace cutf{
namespace math{
MATH_FUNC(ceil);
MATH_FUNC(cos);
MATH_FUNC(exp);
MATH_FUNC(exp10);
MATH_FUNC(exp2);
MATH_FUNC(floor);
MATH_FUNC(log);
MATH_FUNC(log10);
MATH_FUNC(log2);
MATH_FUNC(rcp);
MATH_FUNC(rint);
MATH_FUNC(rsqrt);
MATH_FUNC(sin);
MATH_FUNC(sqrt);
MATH_FUNC(trunc);

// abs
template <class T> T abs(const T a);
template <> CUTF_DEVICE_FUNC inline double abs<double>(const double a){return fabs(a);}
template <> CUTF_DEVICE_FUNC inline float abs<float>(const float a){return fabsf(a);}
template <> CUTF_DEVICE_FUNC inline __half abs<__half>(const __half a){
    const uint16_t abs_a = cutf::experimental::fp::reinterpret_as_uint(a) & 0x7fff;
    return cutf::experimental::fp::reinterpret_as_fp(abs_a);
}
template <> CUTF_DEVICE_FUNC inline __half2 abs<__half2>(const __half2 a){
    const auto abs_a = cutf::experimental::fp::detail::reinterpret_medium<__half2, uint32_t>{.fp = a}.bs & 0x7fff7fff;
		// This reinterpretation can't avoid using `reinterpret_cast` because `__half2` is not a standard numeric type but struct.
		return *reinterpret_cast<const __half2*>(&abs_a);
}
template <> CUTF_DEVICE_FUNC inline int abs<int>(const int a){return ::abs(a);}
template <> CUTF_DEVICE_FUNC inline long int abs<long int>(const long int a){return labs(a);}
template <> CUTF_DEVICE_FUNC inline long long int abs<long long int>(const long long int a){return llabs(a);}

// isinf
template <class T>
CUTF_DEVICE_HOST_FUNC inline bool isnan(const T value) {
	return std::isnan(value);
}
template <>
CUTF_DEVICE_HOST_FUNC inline bool isnan<half>(const half value) {
#ifdef __CUDA_ARCH__
	return __hisnan(value);
#else
	const auto v = cutf::experimental::fp::reinterpret_as_uint(value);
	return (((v >> 10) & 0x1f) == 0x1f) && (v & 0x3ff);
#endif
}

// isnan
template <class T>
CUTF_DEVICE_HOST_FUNC inline bool isinf(const T value) {
	return std::isinf(value);
}
template <>
CUTF_DEVICE_HOST_FUNC inline bool isinf<half>(const half value) {
#ifdef __CUDA_ARCH__
	return __hisinf(value);
#else
	const auto v = cutf::experimental::fp::reinterpret_as_uint(value);
	return (((v >> 10) & 0x1f) == 0x1f) && !(v & 0x3ff);
#endif
}

// get sign
template <class T> CUTF_DEVICE_FUNC inline T sign(const T v);
#ifdef __CUDA_ARCH__
template <> CUTF_DEVICE_FUNC inline double sign(const double v){
	double r;
	asm(R"({
	.reg .b64 %u;
	and.b64 %u, %1, 9223372036854775808;
	or.b64 %0, %u, 4607182418800017408;
})":"=d"(r):"d"(v));
	return r;
}
template <> CUTF_DEVICE_FUNC inline float sign(const float v){
	float r;
	asm(R"({
	.reg .b32 %u;
	and.b32 %u, %1, 2147483648;
	or.b32 %0, %u, 1065353216;
})":"=f"(r):"f"(v));
	return r;
}
#define HALF2CUS(var) *(reinterpret_cast<const unsigned short*>(&(var)))
#define HALF2US(var) *(reinterpret_cast<unsigned short*>(&(var)))
template <> CUTF_DEVICE_FUNC inline half sign(const half v){
	half r;
	asm(R"({
	.reg .b16 %u;
	and.b16 %u, %1, 32768;
	or.b16 %0, %u, 15360;
})":"=h"(HALF2US(r)):"h"(HALF2CUS(v)));
	return r;
}
#define HALF22CUS(var) *(reinterpret_cast<const unsigned int*>(&(var)))
#define HALF22US(var) *(reinterpret_cast<unsigned int*>(&(var)))
template <> CUTF_DEVICE_FUNC inline half2 sign(const half2 v){
	half2 r;
	asm(R"({
	.reg .b32 %u;
	and.b32 %u, %1, 2147516416;
	or.b32 %0, %u, 1006648320;
})":"=r"(HALF22US(r)):"r"(HALF22CUS(v)));
	return r;
}
#endif

#ifdef __CUDA_ARCH__
// max
CUTF_DEVICE_FUNC inline __half2 max(const __half2 a, const __half2 b) {
#if __CUDA_ARCH__ < 800
        const half2 sub = __hsub2(a, b);
        const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
        const unsigned sw = ((sign >> 21) | (sign >> 13)) * 0x11;
        const int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b), 0x00003210 | sw);
        return *reinterpret_cast<const __half2*>(&res);
#else
        return __hmax2(a, b);
#endif
}
CUTF_DEVICE_FUNC inline __half max(const __half a, const __half b) {
#if __CUDA_ARCH__ < 800
        const half sub = __hsub(a, b);
        const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
        const unsigned sw = (sign >> 13) * 0x11;
        const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), *reinterpret_cast<const short*>(&b), 0x00000010 | sw);
        return *reinterpret_cast<const __half*>(&res);
#else
        return __hmax(a, b);
#endif
}
CUTF_DEVICE_FUNC inline float max(const float a, const float b) {return fmaxf(a, b);};
CUTF_DEVICE_FUNC inline double max(const double a, const double b) {return fmax(a, b);};

// For math functions for integers
#define CUTF_MATH_IMAX(bfunc, type_r, type_a, type_b) CUTF_DEVICE_FUNC inline type_r max(const type_a a, const type_b b) {return ::bfunc(a, b);}
CUTF_MATH_IMAX(max   , unsigned long long int, unsigned long long int, unsigned long long int);
CUTF_MATH_IMAX(max   , unsigned long long int, long long int         , unsigned long long int);
CUTF_MATH_IMAX(max   , unsigned long long int, unsigned long long int, long long int         );
CUTF_MATH_IMAX(max   , long long int         , long long int         , long long int         );

CUTF_MATH_IMAX(max   , unsigned long int     , unsigned long int     , unsigned long int     );
CUTF_MATH_IMAX(max   , unsigned long int     , unsigned long int     , long int              );
CUTF_MATH_IMAX(max   , unsigned long int     , long int              , unsigned long int     );
CUTF_MATH_IMAX(max   , long int              , long int              , long int              );

CUTF_MATH_IMAX(umax  , unsigned int          , unsigned int          , unsigned int          );
CUTF_MATH_IMAX(max   , unsigned int          , unsigned int          , int                   );
CUTF_MATH_IMAX(max   , unsigned int          , int                   , unsigned int          );
CUTF_MATH_IMAX(max   , int                   , int                   , int                   );

// min
CUTF_DEVICE_FUNC inline __half2 min(const __half2 a, const __half2 b) {
#if __CUDA_ARCH__ < 800
        const half2 sub = __hsub2(b, a);
        const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
        const unsigned sw = ((sign >> 21) | (sign >> 13)) * 0x11;
        const int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), *reinterpret_cast<const unsigned*>(&b), 0x00003210 | sw);
        return *reinterpret_cast<const __half2*>(&res);
#else
        return __hmin2(a, b);
#endif
}
CUTF_DEVICE_FUNC inline __half min(const __half a, const __half b) {
#if __CUDA_ARCH__ < 800
        const half sub = __hsub(b, a);
        const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
        const unsigned sw = (sign >> 13) * 0x11;
        const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), *reinterpret_cast<const short*>(&b), 0x00000010 | sw);
        return *reinterpret_cast<const __half*>(&res);
#else
        return __hmin(a, b);
#endif
}
CUTF_DEVICE_FUNC inline float min(const float a, const float b) {return fminf(a, b);};
CUTF_DEVICE_FUNC inline double min(const double a, const double b) {return fmin(a, b);};


#define CUTF_MATH_IMIN(bfunc, type_r, type_a, type_b) CUTF_DEVICE_FUNC inline type_r min(const type_a a, const type_b b) {return ::bfunc(a, b);}
CUTF_MATH_IMIN(min   , unsigned long long int, unsigned long long int, unsigned long long int);
CUTF_MATH_IMIN(min   , unsigned long long int, long long int         , unsigned long long int);
CUTF_MATH_IMIN(min   , unsigned long long int, unsigned long long int, long long int         );
CUTF_MATH_IMIN(min   , long long int         , long long int         , long long int         );

CUTF_MATH_IMIN(min   , unsigned long int     , unsigned long int     , unsigned long int     );
CUTF_MATH_IMIN(min   , unsigned long int     , unsigned long int     , long int              );
CUTF_MATH_IMIN(min   , unsigned long int     , long int              , unsigned long int     );
CUTF_MATH_IMIN(min   , long int              , long int              , long int              );

CUTF_MATH_IMIN(umin  , unsigned int          , unsigned int          , unsigned int          );
CUTF_MATH_IMIN(min   , unsigned int          , unsigned int          , int                   );
CUTF_MATH_IMIN(min   , unsigned int          , int                   , unsigned int          );
CUTF_MATH_IMIN(min   , int                   , int                   , int                   );

#else
// prototype
CUTF_DEVICE_FUNC inline __half2 max(const __half2 a, const __half2 b);
CUTF_DEVICE_FUNC inline __half max(const __half a, const __half b);
CUTF_DEVICE_FUNC inline float max(const float a, const float b);
CUTF_DEVICE_FUNC inline double max(const double a, const double b);
CUTF_DEVICE_FUNC inline __half2 min(const __half2 a, const __half2 b);
CUTF_DEVICE_FUNC inline __half min(const __half a, const __half b);
CUTF_DEVICE_FUNC inline float min(const float a, const float b);
CUTF_DEVICE_FUNC inline double min(const double a, const double b);
#endif

namespace horizontal {
#ifdef __CUDA_ARCH__
inline CUTF_DEVICE_HOST_FUNC __half add(const __half2 a) {return a.x + a.y;}
inline CUTF_DEVICE_HOST_FUNC __half mul(const __half2 a) {return a.x * a.y;}
inline CUTF_DEVICE_HOST_FUNC __half max(const __half2 a) {return cutf::math::max(a.x, a.y);}
inline CUTF_DEVICE_HOST_FUNC __half min(const __half2 a) {return cutf::math::min(a.x, a.y);}
#else
inline CUTF_DEVICE_HOST_FUNC __half add(const __half2 a) {return __float2half(__half2float(a.x) + __half2float(a.y));}
inline CUTF_DEVICE_HOST_FUNC __half mul(const __half2 a) {return __float2half(__half2float(a.x) * __half2float(a.y));}
inline CUTF_DEVICE_HOST_FUNC __half max(const __half2 a) {return __float2half(std::max(__half2float(a.x), __half2float(a.y)));}
inline CUTF_DEVICE_HOST_FUNC __half min(const __half2 a) {return __float2half(std::min(__half2float(a.x), __half2float(a.y)));}
#endif
} // namespace horizontal
} // math
} // cutf
#endif // __CUTF_MATH_CUH__
