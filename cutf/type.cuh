#ifndef __CUTF_TYPE_CUH__
#define __CUTF_TYPE_CUH__

#include <cuda_fp16.h>

#define CAST(from_t, to_t, func, val) \
	template <> __host__ __device__ to_t cast<to_t>(const from_t val){return func;}

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

} // namespace type	
} // cuda
} // no name

#endif // __CUTF_TYPE_CUH__
