#ifndef __CUTF_TYPE_CUH__
#define __CUTF_TYPE_CUH__

#include <cuda_fp16.h>
#include <cuComplex.h>
#include "experimental/tf32.hpp"
#include "experimental/fp.hpp"
#include "rounding_mode.hpp"
#include "macro.hpp"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <mma.h>
#define __CUTF_AMPERE_MMA__
#else
struct __nv_bfloat16;
namespace nvcuda {
namespace wmma {
namespace precision {
struct tf32;
} // precision
} // nvcuda
} // nvcuda
#endif

#define CAST(from_t, to_t, func, val) \
	 template <> CUTF_DEVICE_HOST_FUNC inline typename data_t<to_t>::type cast<to_t>(const from_t val){return func;}

#ifdef __CUDA_ARCH__
#define REINTERPRET(src_type, src_ty, dst_type, dst_ty) \
	 template <> CUTF_DEVICE_FUNC inline dst_type reinterpret<dst_type>(const src_type a){return __##src_ty##_as_##dst_ty(a);}
#else
#define REINTERPRET(src_type, src_ty, dst_type, dst_ty) \
	 template <> CUTF_DEVICE_FUNC inline dst_type reinterpret<dst_type>(const src_type a){return cutf::experimental::fp::detail::reinterpret_medium<src_type, dst_type>{.fp = a}.bs;}
#endif

#ifdef __CUDA_ARCH__
#define RCAST(src_type, src_ty, dst_type, dst_ty, r) \
	 template <> CUTF_DEVICE_FUNC inline dst_type rcast<dst_type, cutf::rounding::r>(const src_type a){return __##src_ty##2##dst_ty##_##r(a);}
#else
#define RCAST(src_type, src_ty, dst_type, dst_ty, r) \
	 template <> CUTF_DEVICE_FUNC inline dst_type rcast<dst_type, cutf::rounding::r>(const src_type a){return static_cast<dst_type>(a);}
#endif
#define RCASTS(src_type, src_ty, dst_type, dst_ty) \
	RCAST(src_type, src_ty, dst_type, dst_ty, rd); \
	RCAST(src_type, src_ty, dst_type, dst_ty, rn); \
	RCAST(src_type, src_ty, dst_type, dst_ty, ru); \
	RCAST(src_type, src_ty, dst_type, dst_ty, rz);

namespace cutf{
namespace type {
template <class T>
struct data_t {using type = T;};
template <> struct data_t<nvcuda::wmma::precision::tf32> {using type = float;};

template <class T>  CUTF_DEVICE_HOST_FUNC inline typename data_t<T>::type cast(const int a)    {return static_cast<T>(a);};
template <class T>  CUTF_DEVICE_HOST_FUNC inline typename data_t<T>::type cast(const half a)   {return static_cast<T>(a);};
template <class T>  CUTF_DEVICE_HOST_FUNC inline typename data_t<T>::type cast(const float a)  {return static_cast<T>(a);};
template <class T>  CUTF_DEVICE_HOST_FUNC inline typename data_t<T>::type cast(const double a) {return static_cast<T>(a);};


#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
CAST(int, half, __float2half(static_cast<float>(a)), a);

CAST(half  , float , __half2float(a), a);
CAST(half  , double, static_cast<double>(__half2float(a)), a);
CAST(float , half  , __float2half(a), a);
CAST(double, half  , __half2float(static_cast<float>(a)), a);
#endif

// cast to tf32
template <>  CUTF_DEVICE_HOST_FUNC inline typename data_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const int a) {
#if defined(__CUTF_AMPERE_MMA__)
    float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(cutf::type::cast<float>(a)));
    return ret;
#else
	return cutf::experimental::tf32::to_tf32(cutf::type::cast<float>(a));
#endif
}
template <>  CUTF_DEVICE_HOST_FUNC inline typename data_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const half a) {
#if defined(__CUTF_AMPERE_MMA__)
    float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(cutf::type::cast<float>(a)));
    return ret;
#else
	return cutf::experimental::tf32::to_tf32(cutf::type::cast<float>(a));
#endif
}
template <>  CUTF_DEVICE_HOST_FUNC inline typename data_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const float a) {
#if defined(__CUTF_AMPERE_MMA__)
    float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(cutf::type::cast<float>(a)));
    return ret;
#else
	return cutf::experimental::tf32::to_tf32(cutf::type::cast<float>(a));
#endif
}
template <>  CUTF_DEVICE_HOST_FUNC inline typename data_t<nvcuda::wmma::precision::tf32>::type cast<nvcuda::wmma::precision::tf32>(const double a) {
#if defined(__CUTF_AMPERE_MMA__)
    float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(cutf::type::cast<float>(a)));
    return ret;
#else
	return cutf::experimental::tf32::to_tf32(cutf::type::cast<float>(a));
#endif
}

// reinterpret
template <class T>  CUTF_DEVICE_FUNC inline T reinterpret(const float a);
template <class T>  CUTF_DEVICE_FUNC inline T reinterpret(const double a);
template <class T>  CUTF_DEVICE_FUNC inline T reinterpret(const long long a);
template <class T>  CUTF_DEVICE_FUNC inline T reinterpret(const unsigned int a);
template <class T>  CUTF_DEVICE_FUNC inline T reinterpret(const int a);
REINTERPRET(float, float, unsigned int, uint);
REINTERPRET(float, float, int, int);
REINTERPRET(double, double, long long, longlong);
REINTERPRET(int, int, float, float);
REINTERPRET(unsigned int, uint, float, float);
REINTERPRET(long long, longlong, double, double);

// rounding cast
template <class T, class R>  CUTF_DEVICE_FUNC inline T rcast(const float a);
template <class T, class R>  CUTF_DEVICE_FUNC inline T rcast(const double a);
template <class T, class R>  CUTF_DEVICE_FUNC inline T rcast(const int a);
template <class T, class R>  CUTF_DEVICE_FUNC inline T rcast(const unsigned int a);
template <class T, class R>  CUTF_DEVICE_FUNC inline T rcast(const unsigned long long int a);
template <class T, class R>  CUTF_DEVICE_FUNC inline T rcast(const long long int a);

RCASTS(float, float, int, int);
RCASTS(float, float, long long int, ll);
RCASTS(float, float, unsigned int, uint);
RCASTS(float, float, unsigned long long int, ull);
RCASTS(double, double, float, float);
RCASTS(double, double, int, int);
RCASTS(double, double, long long int, ll);
RCASTS(double, double, unsigned int, uint);
RCASTS(double, double, unsigned long long int, ull);
RCASTS(int, int, float, float);
RCASTS(long long int , ll, double, double);
RCASTS(long long int , ll, float, float);
RCASTS(unsigned int , uint, float, float);
RCASTS(unsigned long long int , ull, double, double);
RCASTS(unsigned long long int , ull, float, float);

#define DATA_TYPE_DEF(type_name, number_type, type_size) \
	template <> inline cudaDataType_t get_data_type<type_name>(){return CUDA_##number_type##_##type_size;}
template <class T>
inline cudaDataType_t get_data_type();
DATA_TYPE_DEF(half, R, 16F);
DATA_TYPE_DEF(half2, C, 16F);
DATA_TYPE_DEF(float, R, 32F);
DATA_TYPE_DEF(cuComplex, C, 32F);
DATA_TYPE_DEF(double, R, 64F);
DATA_TYPE_DEF(cuDoubleComplex, C, 64F);
// Uncertain {{{
DATA_TYPE_DEF(char, R, 8I);
DATA_TYPE_DEF(short, C, 8I);
DATA_TYPE_DEF(unsigned char, R, 8U);
DATA_TYPE_DEF(unsigned short, C, 8U);
// }}}
} // namespace type	
} // cutf

#endif // __CUTF_TYPE_CUH__
