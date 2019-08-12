#ifndef __CUTF_TYPE_CUH__
#define __CUTF_TYPE_CUH__

#include <cuda_fp16.h>
#include <cuComplex.h>

#define CAST(from_t, to_t, func, val) \
	 template <> __host__ __device__ inline to_t cast<to_t>(const from_t val){return func;}
#define REINTERPRET(src_type, src_ty, dst_type, dst_ty) \
	 template <> __device__ inline dst_type reinterpret<dst_type>(const src_type a){return __##src_ty##_as_##dst_ty(a);}

#define RCAST(src_type, src_ty, dst_type, dst_ty, r) \
	 template <> __device__ inline dst_type rcast<dst_type, rounding::r>(const src_type a){return __##src_ty##2##dst_ty##_##r(a);}
#define RCASTS(src_type, src_ty, dst_type, dst_ty) \
	RCAST(src_type, src_ty, dst_type, dst_ty, rd); \
	RCAST(src_type, src_ty, dst_type, dst_ty, rn); \
	RCAST(src_type, src_ty, dst_type, dst_ty, ru); \
	RCAST(src_type, src_ty, dst_type, dst_ty, rz); 

namespace cutf{
namespace type {
template <class T>  __host__ __device__ inline T cast(const int a);
template <class T>  __host__ __device__ inline T cast(const half a);
template <class T>  __host__ __device__ inline T cast(const float a);
template <class T>  __host__ __device__ inline T cast(const double a);

CAST(int, int, a, a);
CAST(int, half, __float2half(static_cast<float>(a)), a);
CAST(int, float, static_cast<float>(a), a);
CAST(int, double, static_cast<double>(a), a);

CAST(half, int, static_cast<int>(__half2float(a)), a);
CAST(half, half, a, a);
CAST(half, float, __half2float(a), a);
CAST(half, double, static_cast<double>(__half2float(a)), a);

CAST(float, int, static_cast<int>(a), a);
CAST(float, half, __float2half(a), a);
CAST(float, float, a, a);
CAST(float, double, static_cast<double>(a), a);

CAST(double, int, static_cast<int>(a), a);
CAST(double, half, __half2float(static_cast<float>(a)), a);
CAST(double, float, static_cast<float>(a), a);
CAST(double, double, a, a);

// reinterpret
template <class T>  __device__ inline T reinterpret(const float a);
template <class T>  __device__ inline T reinterpret(const double a);
template <class T>  __device__ inline T reinterpret(const long long a);
template <class T>  __device__ inline T reinterpret(const unsigned int a);
template <class T>  __device__ inline T reinterpret(const int a);
REINTERPRET(float, float, unsigned int, uint);
REINTERPRET(float, float, int, int);
REINTERPRET(double, double, long long, longlong);
REINTERPRET(int, int, float, float);
REINTERPRET(unsigned int, uint, float, float);
REINTERPRET(long long, longlong, double, double);

// rounding cast
namespace rounding{
	struct rd;
	struct rn;
	struct ru;
	struct rz;
};
template <class T, class R>  __device__ inline T rcast(const float a);
template <class T, class R>  __device__ inline T rcast(const double a);
template <class T, class R>  __device__ inline T rcast(const int a);
template <class T, class R>  __device__ inline T rcast(const unsigned int a);
template <class T, class R>  __device__ inline T rcast(const unsigned long long int a);
template <class T, class R>  __device__ inline T rcast(const long long int a);

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
