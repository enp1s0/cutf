#ifndef __CUTF_CUBLAS_CUH__
#define __CUTF_CUBLAS_CUH__
#include <cublas_v2.h>

namespace mtk{
namespace cublas{
// amax
#define AMAX_DEF(type_name, short_type_name)\
	inline cublasStatus_t iamax(cublasHandle_t handle, int n, const type_name* x, int incx, int *result) {\
		return cublasI##type_short_name##amax(handle, n, x, incx, result);\
	}
AMAX_DEF(float, s);
AMAX_DEF(double, d);
AMAX_DEF(cuComplex, c);
AMAX_DEF(cuDoubleComplex, z);

// amin
#define AMIN_DEF(type_name, short_type_name)\
	inline cublasStatus_t iamin(cublasHandle_t handle, int n, const type_name* x, int incx, int *result) {\
		return cublasI##type_short_name##amin(handle, n, x, incx, result);\
	}
AMIN_DEF(float, s);
AMIN_DEF(double, d);
AMIN_DEF(cuComplex, c);
AMIN_DEF(cuDoubleComplex, z);

// asum
#define ASUM_DEF(type_name, short_type_name, result_type_name)\
	inline cublasStatus_t asum(cublasHandle_t handle, int n, const type_name* x, int incx, result_type_name *result) {\
		return cublas##type_short_name##asum(handle, n, x, incx, result);\
	}
ASUM_DEF(float, S, float);
ASUM_DEF(double, D, double);
ASUM_DEF(cuComplex, Sc, float);
ASUM_DEF(cuDoubleComplex, Dz, double);

// axpy
#define AXPY_DEF(type_name, short_type_name)\
	inline cublasStatus_t axpy(cublasHandle_t handle, int n, const type_name *alpha, const type_name *x, int incx, type_name *y, int incy) {\
		return cublas##short_type_name##axpy(handle, n, alpha, x, incx, y, incy); \
	}
AXPY_DEF(float, S);
AXPY_DEF(double, D);
AXPY_DEF(cuComplex, C);
AXPY_DEF(cuDoubleComplex, Z);

// copy
#define COPY_DEF(type_name, short_type_name)\
	inline cublasStatus_t copy(cublasHandle_t handle, int n, const type_name *x, int incx, type_name *y, int incy) {\
		return cublas##short_type_name##copy(handle, n, x, incx, y, incy); \
	}
COPY_DEF(float, S);
COPY_DEF(double, D);
COPY_DEF(cuComplex, C);
COPY_DEF(cuDoubleComplex, Z);

// dot
#define DOT_DEF(type_name, short_type_name)\
	inline cublasStatus_t dot(cublasHandle_t handle, int n, const type_name* x, int incx, const type_name* y, int incy, type_name *result) {\
		return cublas##type_short_name##dot(handle, n, x, incx, result);\
	}
#define DOT_DEF(type_name, short_type_name, uc)\
	inline cublasStatus_t dot##uc(cublasHandle_t handle, int n, const type_name* x, int incx, const type_name* y, int incy, type_name *result) {\
		return cublas##type_short_name##dot##uc(handle, n, x, incx, y, incy, result);\
	}
DOT_DEF(float, S);
DOT_DEF(double, D);
DOT_DEF(cuComplex, C, u);
DOT_DEF(cuDoubleComplex, Z, u);
DOT_DEF(cuComplex, C, c);
DOT_DEF(cuDoubleComplex, Z, c);

// nrm2
#define NRM2_DEF(type_name, short_type_name, result_type_name)\
	inline cublasStatus_t nrm2(cublasHandle_t handle, int n, const type_name *x, int incx, result_type_name* result) {\
		return cublas##short_type_name##nrm2(handle, n, x, incx, result); \
	}
NRM2_DEF(float, S, float);
NRM2_DEF(double, D, double);
NRM2_DEF(cuComplex, Sc, float);
NRM2_DEF(cuDoubleComplex, Dz, double);

// nrm2
#define ROT_DEF(type_name, short_type_name, cosine_type_name, result_type_name)\
	inline cublasStatus_t rot(cublasHandle_t handle, int n, const type_name *x, int incx, const type_name *y, int incy, const cosine_type_name* c, const sine_type_name* s) {\
		return cublas##short_type_name##rot(handle, n, x, incx, y, incy, c, s); \
	}
ROT_DEF(float, S, float, float);
ROT_DEF(double, D, double, double);
ROT_DEF(cuComplex, C, float, cuComplex);
ROT_DEF(cuComplex, Cs, float, float);
ROT_DEF(cuDoubleComplex, Z, double, cuDoubleComplex);
ROT_DEF(cuDoubleComplex, Zd, double, double);

// rotg
#define ROTG_DEF(type_name, cosine_type_name)\
	inline cublasStatus_t rotg(cublasHandle_t handle, type_name *a, type_name* b, cosine_type_name* c, type_name *s){\
		return cublas##short_type_name##rotg(handle, a, b, c, s); \
	}
NRM2_DEF(float, S, float);
NRM2_DEF(double, D, double);
NRM2_DEF(cuComplex, C, float);
NRM2_DEF(cuDoubleComplex, Z, double);

// rotm
#define ROTM_DEF(type_name, short_type_name)\
	inline cublasStatus_t rotm(cublasHandle_t handle, int n, type_name* x, int incx, type_name* y, int incy, const type_name* param){\
		return cublas##short_type_name##rotm(handle, n, x, incx, y, incy, param); \
	}
ROTM_DEF(float, S);
ROTM_DEF(double, D);

// rotmg
#define ROTMG_DEF(type_name, short_type_name)\
	inline cublasStatus_t rotm(cublasHandle_t handle, type_name* d1, type_name* d2, type_name* x1, const type_name* y1, type_name* param){\
		return cublas##short_type_name##rotmg(handle, d1, d2, x1, y1, param); \
	}
ROTMG_DEF(float, S);
ROTMG_DEF(double, D);

// scale
#define SCALE_DEF(type_name, short_type_name, scale_type_name)\
	inline cublasStatus_t scale(cublasHandle_t handle, int n, const scale_type_name* alpha, type_name* x, int incx){\
		return cublas##short_type_name##scale(handle, n, alpha, x, incx);
	}

SCALE_DEF(float, S, float);
SCALE_DEF(double, S, double);
SCALE_DEF(cuComplex, C, cuComplex);
SCALE_DEF(cuComplex, Cs, float);
SCALE_DEF(cuDoubleComplex, Z, cuDoubleComplex);
SCALE_DEF(cuDoubleComplex, Zd, double);

// swap
#define SWAP_DEF(type_name, short_type_name)\
	inline cublasStatus_t swap(cublasHandle_t handle, int n, type_name* x, int incx, type_name* y, int incy){ \
		return cublas##short_type_name##swap(handle, n, x, incx, y, incy);
	}
SWAP_DEF(float, S);
SWAP_DEF(double, D);
SWAP_DEF(cuComplex, C);
SWAP_DEF(cuDoubleComplex, Z);

} // cublas
} // mtk

#endif // __CUTF_CUBLAS_CUH__
