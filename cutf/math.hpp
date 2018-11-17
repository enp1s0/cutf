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
namespace cuda{
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
} // math
} // cuda
} // cutf
#endif // __CUTF_MATH_CUH__
