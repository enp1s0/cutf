#ifndef __CUTF_TYPE_CUH__
#define __CUTF_TYPE_CUH__

#include <cuda_fp16.h>

#define CAST(from_t, to_t, func, val) \
	template <> __host__ __device__ to_t cast<to_t>(const from_t val){return func;}
#define REINTERPRET(src_type, src_ty, dst_type, dst_ty) \
	template <> __device__ dst_type reinterpret<dst_type>(const src_type a){return __##src_ty##_as_##dst_ty(a);}

namespace{
namespace cuda {
namespace type {
template <class T> __host__ __device__ T cast(const half a);
template <class T> __host__ __device__ T cast(const float a);
template <class T> __host__ __device__ T cast(const double a);

CAST(half, half, a, a);
CAST(half, float, __half2float(a), a);
CAST(half, double, static_cast<double>(__half2float(a)), a);

CAST(float, half, __float2half(a), a);
CAST(float, float, a, a);
CAST(float, double, static_cast<double>(a), a);

CAST(double, half, __half2float(static_cast<float>(a)), a);
CAST(double, float, static_cast<float>(a), a);
CAST(double, double, a, a);

// reinterpret
template <class T> __device__ T reinterpret(const float a);
template <class T> __device__ T reinterpret(const double a);
template <class T> __device__ T reinterpret(const long long a);
template <class T> __device__ T reinterpret(const unsigned int a);
template <class T> __device__ T reinterpret(const int a);
REINTERPRET(float, float, unsigned int, uint);
REINTERPRET(float, float, int, int);
REINTERPRET(double, double, long long, longlong);
REINTERPRET(int, int, float, float);
REINTERPRET(unsigned int, uint, float, float);
REINTERPRET(long long, longlong, double, double);

} // namespace type	
} // cuda
} // no name

#endif // __CUTF_TYPE_CUH__
