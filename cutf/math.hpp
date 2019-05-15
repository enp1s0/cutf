#ifndef __CUTF_MATH_CUH__
#define __CUTF_MATH_CUH__

#include <cuda_fp16.h>

#define DEF_TEMPLATE_MATH_FUNC_1(func) \
template<class T>  __device__ inline T func(const T a);

#define SPEC_MATH_FUNC_1_h( func ) \
template<>  __device__ inline half func<half>(const half a){return h##func( a );}	
#define SPEC_MATH_FUNC_1_h2( func ) \
template<>  __device__ inline half2 func<half2>(const half2 a){return h2##func( a );}	
#define SPEC_MATH_FUNC_1_f( func ) \
template<>  __device__ inline float func<float>(const float a){return func##f( a );}	
#define SPEC_MATH_FUNC_1_d( func ) \
template<>  __device__ inline double func<double>(const double a){return func( a );}	

#define MATH_FUNC(func) \
	DEF_TEMPLATE_MATH_FUNC_1(func) \
	SPEC_MATH_FUNC_1_h(func) \
	SPEC_MATH_FUNC_1_h2(func) \
	SPEC_MATH_FUNC_1_f(func) \
	SPEC_MATH_FUNC_1_d(func) \


 __device__ inline float rcpf(const float a){return __frcp_rn(a);}
 __device__ inline double rcp(const double a){return __drcp_rn(a);}
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

// get sign
template <class T> __device__ inline T sign(const T v);
template <> __device__ inline double sign(const double v){
	double r;
	asm(R"({
	.reg .b64 %u;
	and.b64 %u, %1, 9223372036854775808;
	or.b64 %0, %u, 4607182418800017408;
})":"=d"(r):"d"(v));
	return r;
}
template <> __device__ inline float sign(const float v){
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
template <> __device__ inline half sign(const half v){
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
template <> __device__ inline half2 sign(const half2 v){
	half2 r;
	asm(R"({
	.reg .b32 %u;
	and.b32 %u, %1, 2147516416;
	or.b32 %0, %u, 1006648320;
})":"=r"(HALF22US(r)):"r"(HALF22CUS(v)));
	return r;
}
} // math
} // cutf
#endif // __CUTF_MATH_CUH__
