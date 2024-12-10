#ifndef __CUTF_TYPE_CUH__
#define __CUTF_TYPE_CUH__

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuComplex.h>
#include "experimental/tf32.hpp"
#include "experimental/fp.hpp"
#include "rounding_mode.hpp"
#include "macro.hpp"

#ifdef __CUTF_FP8_EXIST__
#include <cuda_fp8.h>
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <mma.h>
#define __CUTF_AMPERE_MMA__
#else
namespace nvcuda {
namespace wmma {
namespace precision {
struct tf32;
} // precision
} // wmma
} // nvcuda
#endif
#ifndef __CUDA_BF16_TYPES_EXIST__
struct __CUDA_ALIGN__(2) __nv_bfloat16 {
    unsigned short __x;
};
struct __CUDA_ALIGN__(4) __nv_bfloat162 {
    unsigned __x;
};
#endif

#ifdef __CUDA_ARCH__
#define REINTERPRET(src_type, src_ty, dst_type, dst_ty) \
	 template <> CUTF_DEVICE_HOST_FUNC inline dst_type reinterpret<dst_type>(const src_type a){return __##src_ty##_as_##dst_ty(a);}
#else
#define REINTERPRET(src_type, src_ty, dst_type, dst_ty) \
	 template <> CUTF_DEVICE_HOST_FUNC inline dst_type reinterpret<dst_type>(const src_type a){return cutf::experimental::fp::detail::reinterpret_medium<src_type, dst_type>{.fp = a}.bs;}
#endif

#ifdef __CUDA_ARCH__
#define RCAST(src_type, src_ty, dst_type, dst_ty, r) \
	 template <> CUTF_DEVICE_HOST_FUNC inline dst_type rcast<dst_type, cutf::rounding::r>(const src_type a){return __##src_ty##2##dst_ty##_##r(a);}
#else
#define RCAST(src_type, src_ty, dst_type, dst_ty, r) \
	 template <> CUTF_DEVICE_HOST_FUNC inline dst_type rcast<dst_type, cutf::rounding::r>(const src_type a){return static_cast<dst_type>(a);}
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

namespace detail {
template <class DST_T, class SRC_T>
struct cast_op {
  inline __device__ typename data_t<DST_T>::type operator()(const SRC_T a) {
    return static_cast<typename data_t<DST_T>::type>(a);
  }
};

template <class SRC_T>
struct cast_op<nvcuda::wmma::precision::tf32, SRC_T> {
  inline __device__ typename data_t<nvcuda::wmma::precision::tf32>::type operator()(const SRC_T a) {
    const auto v = cutf::type::detail::cast_op<float, SRC_T>{}(a);
#if defined(__CUTF_AMPERE_MMA__)
    float ret;
    asm("{.reg .b32 %mr;\n"
        "cvt.rna.tf32.f32 %mr, %1;\n"
        "mov.b32 %0, %mr;}\n" : "=f"(ret) : "f"(v));
    return ret;
#else
	return cutf::experimental::tf32::to_tf32(v);
#endif
  }
};

template <>
struct cast_op<cuDoubleComplex, cuComplex> {
  inline __device__ typename data_t<cuDoubleComplex>::type operator()(const cuComplex a) {
    return make_cuDoubleComplex(a.x, a.y);
  }
};
template <>
struct cast_op<cuComplex, cuComplex> {
  inline __device__ typename data_t<cuComplex>::type operator()(const cuComplex a) {
    return make_cuComplex(a.x, a.y);
  }
};
template <>
struct cast_op<cuComplex, cuDoubleComplex> {
  inline __device__ typename data_t<cuComplex>::type operator()(const cuDoubleComplex a) {
    return make_cuComplex(a.x, a.y);
  }
};
template <>
struct cast_op<cuDoubleComplex, cuDoubleComplex> {
  inline __device__ typename data_t<cuDoubleComplex>::type operator()(const cuDoubleComplex a) {
    return make_cuDoubleComplex(a.x, a.y);
  }
};
} // namespace detail

// cast
template <class T, class S>  CUTF_DEVICE_HOST_FUNC inline typename data_t<T>::type cast(const S a)   {return detail::cast_op<T, S>{}(a);}

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
template <> constexpr cudaDataType_t get_data_type<type_name>(){return CUDA_##number_type##_##type_size;}
template <class T>
constexpr cudaDataType_t get_data_type();
DATA_TYPE_DEF(half, R, 16F);
DATA_TYPE_DEF(half2, C, 16F);
DATA_TYPE_DEF(__nv_bfloat16, R, 16BF);
DATA_TYPE_DEF(__nv_bfloat162, C, 16BF);
DATA_TYPE_DEF(float, R, 32F);
DATA_TYPE_DEF(cuComplex, C, 32F);
DATA_TYPE_DEF(double, R, 64F);
DATA_TYPE_DEF(cuDoubleComplex, C, 64F);
#ifdef __CUTF_FP8_EXIST__
DATA_TYPE_DEF(__nv_fp8_e5m2, R, 8F_E5M2);
DATA_TYPE_DEF(__nv_fp8_e4m3, R, 8F_E4M3);
#endif
// Uncertain {{{
DATA_TYPE_DEF(signed char, R, 8I);
DATA_TYPE_DEF(signed short, R, 16I);
DATA_TYPE_DEF(signed int, R, 32I);
DATA_TYPE_DEF(signed long, R, 64I);
DATA_TYPE_DEF(unsigned char, R, 8U);
DATA_TYPE_DEF(unsigned short, R, 16U);
DATA_TYPE_DEF(unsigned int, R, 32U);
DATA_TYPE_DEF(unsigned long, R, 64U);
// }}}

// Complex
template <class T>
struct real_type {using type = T;};
template <> struct real_type<cuDoubleComplex> {using type = double;};
template <> struct real_type<cuComplex      > {using type = float;};
template <class T>
struct complex_type {using type = T;};
template <> struct complex_type<double> {using type = cuDoubleComplex;};
template <> struct complex_type<float > {using type = cuComplex;};

template <class COMPLEX_T, class REAL_T>
CUTF_DEVICE_HOST_FUNC COMPLEX_T to_complex(const REAL_T r) {
	COMPLEX_T c;
	c.x = cast<typename real_type<COMPLEX_T>::type>(r);
	c.y = 0;
	return c;
}

template <class COMPLEX_T>
CUTF_DEVICE_HOST_FUNC COMPLEX_T make_complex(
		const typename real_type<COMPLEX_T>::type x,
		const typename real_type<COMPLEX_T>::type y
		) {
	COMPLEX_T c;
	c.x = x;
	c.y = y;

	return c;
}
} // namespace type	
} // cutf

#endif // __CUTF_TYPE_CUH__
