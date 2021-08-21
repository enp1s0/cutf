#ifndef __CUTF_CUBLAS_CUH__
#define __CUTF_CUBLAS_CUH__
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <sstream>
#include <memory>
#include "error.hpp"
#include "cuda.hpp"

namespace cutf{
namespace error{
inline void check(cublasStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUBLAS_STATUS_SUCCESS){
		std::string error_string;
#define CUBLAS_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_SUCCESS );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_NOT_INITIALIZED );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_ALLOC_FAILED );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_INVALID_VALUE );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_ARCH_MISMATCH );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_MAPPING_ERROR );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_EXECUTION_FAILED );
			CUBLAS_ERROR_CASE( CUBLAS_STATUS_INTERNAL_ERROR );
		default: error_string = "Unknown error"; break;
		}
		std::stringstream ss;
		ss<< error_string;
		if(message.length() != 0){
			ss<<" : "<<message;
		}
	    ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
} // error
namespace cublas{
struct cublas_deleter{
	void operator()(cublasHandle_t* handle){
		cutf::error::check(cublasDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
inline std::unique_ptr<cublasHandle_t, cublas_deleter> get_cublas_unique_ptr(){
	cublasHandle_t *handle = new cublasHandle_t;
	cublasCreate(handle);
	return std::unique_ptr<cublasHandle_t, cublas_deleter>{handle};
}
// ==================================================
// BLAS Lv 1
// ==================================================
// amax
#define AMAX_DEF(type_name, short_type_name)\
	inline cublasStatus_t iamax(cublasHandle_t handle, int n, const type_name* x, int incx, int *result) {\
		return cublasI##short_type_name##amax(handle, n, x, incx, result);\
	}
AMAX_DEF(float, s);
AMAX_DEF(double, d);
AMAX_DEF(cuComplex, c);
AMAX_DEF(cuDoubleComplex, z);

// amin
#define AMIN_DEF(type_name, short_type_name)\
	inline cublasStatus_t iamin(cublasHandle_t handle, int n, const type_name* x, int incx, int *result) {\
		return cublasI##short_type_name##amin(handle, n, x, incx, result);\
	}
AMIN_DEF(float, s);
AMIN_DEF(double, d);
AMIN_DEF(cuComplex, c);
AMIN_DEF(cuDoubleComplex, z);

// asum
#define ASUM_DEF(type_name, short_type_name, result_type_name)\
	inline cublasStatus_t asum(cublasHandle_t handle, int n, const type_name* x, int incx, result_type_name *result) {\
		return cublas##short_type_name##asum(handle, n, x, incx, result);\
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
		return cublas##short_type_name##dot(handle, n, x, incx, y, incy, result);\
	}
#define DOT_DEF_UC(type_name, short_type_name, uc)\
	inline cublasStatus_t dot##uc(cublasHandle_t handle, int n, const type_name* x, int incx, const type_name* y, int incy, type_name *result) {\
		return cublas##short_type_name##dot##uc(handle, n, x, incx, y, incy, result);\
	}
DOT_DEF(float, S);
DOT_DEF(double, D);
DOT_DEF_UC(cuComplex, C, u);
DOT_DEF_UC(cuDoubleComplex, Z, u);
DOT_DEF_UC(cuComplex, C, c);
DOT_DEF_UC(cuDoubleComplex, Z, c);

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
#define ROT_DEF(type_name, short_type_name, cosine_type_name, sine_type_name)\
	inline cublasStatus_t rot(cublasHandle_t handle, int n, type_name *x, int incx, type_name *y, int incy, const cosine_type_name* c, const sine_type_name* s) {\
		return cublas##short_type_name##rot(handle, n, x, incx, y, incy, c, s); \
	}
ROT_DEF(float, S, float, float);
ROT_DEF(double, D, double, double);
ROT_DEF(cuComplex, C, float, cuComplex);
ROT_DEF(cuComplex, Cs, float, float);
ROT_DEF(cuDoubleComplex, Z, double, cuDoubleComplex);
ROT_DEF(cuDoubleComplex, Zd, double, double);

// rotg
#define ROTG_DEF(type_name, short_type_name, cosine_type_name)\
	inline cublasStatus_t rotg(cublasHandle_t handle, type_name *a, type_name* b, cosine_type_name* c, type_name *s){\
		return cublas##short_type_name##rotg(handle, a, b, c, s); \
	}
ROTG_DEF(float, S, float);
ROTG_DEF(double, D, double);
ROTG_DEF(cuComplex, C, float);
ROTG_DEF(cuDoubleComplex, Z, double);

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
	inline cublasStatus_t scal(cublasHandle_t handle, int n, const scale_type_name* alpha, type_name* x, int incx){\
		return cublas##short_type_name##scal(handle, n, alpha, x, incx); \
	}

SCALE_DEF(float, S, float);
SCALE_DEF(double, D, double);
SCALE_DEF(cuComplex, C, cuComplex);
SCALE_DEF(cuComplex, Cs, float);
SCALE_DEF(cuDoubleComplex, Z, cuDoubleComplex);
SCALE_DEF(cuDoubleComplex, Zd, double);

// swap
#define SWAP_DEF(type_name, short_type_name)\
	inline cublasStatus_t swap(cublasHandle_t handle, int n, type_name* x, int incx, type_name* y, int incy){ \
		return cublas##short_type_name##swap(handle, n, x, incx, y, incy); \
	}
SWAP_DEF(float, S);
SWAP_DEF(double, D);
SWAP_DEF(cuComplex, C);
SWAP_DEF(cuDoubleComplex, Z);

// ==================================================
// BLAS Lv 2
// ==================================================
// gbmv
#define GBMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t gbmv(cublasHandle_t handle, \
			cublasOperation_t trans, \
			int m, int n, int kl, int ku, \
			const type_name* alpha, \
			const type_name* A, int lda, \
			const type_name* x, int incx, \
			const type_name* beta, \
			type_name* y, int incy) {\
		return cublas##short_type_name##gbmv(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy); \
	}
GBMV_DEF(float, S);
GBMV_DEF(double, D);
GBMV_DEF(cuComplex, C);
GBMV_DEF(cuDoubleComplex, Z);

// gemv
#define GEMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t gbmv(cublasHandle_t handle, \
			cublasOperation_t trans, \
			int m, int n, \
			const type_name* alpha, \
			const type_name* A, int lda, \
			const type_name* x, int incx, \
			const type_name* beta, \
			type_name* y, int incy) {\
		return cublas##short_type_name##gemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy); \
	}
GEMV_DEF(float, S);
GEMV_DEF(double, D);
GEMV_DEF(cuComplex, C);
GEMV_DEF(cuDoubleComplex, Z);

// ger
#define GER_DEF(type_name, short_type_name) \
	inline cublasStatus_t ger(cublasHandle_t handle, \
		   	int m, int n,\
			const type_name *alpha,\
			const type_name *x, int incx,\
			const type_name *y, int incy,\
			type_name *A, int lda){ \
		return cublas##short_type_name##ger(handle, m, n, alpha, x, incx, y, incy, A, lda);\
	}
#define GER_DEF_UC(type_name, short_type_name, cu) \
	inline cublasStatus_t ger##cu(cublasHandle_t handle, \
		   	int m, int n,\
			const type_name *alpha,\
			const type_name *x, int incx,\
			const type_name *y, int incy,\
			type_name *A, int lda){ \
		return cublas##short_type_name##ger##cu(handle, m, n, alpha, x, incx, y, incy, A, lda);\
	}
GER_DEF(float, S);
GER_DEF(double, D);
GER_DEF_UC(cuComplex, C, c);
GER_DEF_UC(cuComplex, C, u);
GER_DEF_UC(cuDoubleComplex, Z, c);
GER_DEF_UC(cuDoubleComplex, Z, u);

// sbmv
#define SBMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t sbmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, int k, const type_name  *alpha, \
			const type_name *A, int lda, \
			const type_name *x, int incx, \
			const type_name *beta, type_name*y, int incy){ \
		return cublas##short_type_name##sbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy); \
	}
SBMV_DEF(float, S);
SBMV_DEF(double, D);

// spmv
#define SPMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t spmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name  *alpha, const type_name  *AP, \
			const type_name  *x, int incx, const type_name  *beta, \
			type_name  *y, int incy){ \
		return cublas##short_type_name##spmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy); \
	}
SPMV_DEF(float, S);
SPMV_DEF(double, D);

// spr
#define SPR_DEF(type_name, short_type_name) \
	inline cublasStatus_t spr(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name  *alpha, \
			const type_name  *x, int incx, type_name  *AP){ \
		return cublas##short_type_name##spr(handle, uplo, n, alpha, x, incx, AP); \
	}
SPR_DEF(float, S);
SPR_DEF(double, D);

// spr2
#define SPR2_DEF(type_name, short_type_name) \
	inline cublasStatus_t spr2(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name  *alpha, \
			const type_name *x, int incx, \
			const type_name *y, int incy, type_name  *AP){ \
		return cublas##short_type_name##spr2(handle, uplo, n, alpha, x, incx, y, incy, AP); \
	}
SPR2_DEF(float, S);
SPR2_DEF(double, D);

// symv
#define SYMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t symv(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *x, int incx, const type_name *beta, \
			type_name *y, int incy) {\
		return cublas##short_type_name##symv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy); \
	}
SYMV_DEF(float, S);
SYMV_DEF(double, D);
SYMV_DEF(cuComplex, C);
SYMV_DEF(cuDoubleComplex, Z);

// syr
#define SYR_DEF(type_name, short_type_name) \
	inline cublasStatus_t syr(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name *alpha, \
			const type_name *x, int incx, type_name *A, int lda){\
		return cublas##short_type_name##syr(handle, uplo, n, alpha, x, incx, A, lda); \
	}
SYR_DEF(float, S);
SYR_DEF(double, D);
SYR_DEF(cuComplex, C);
SYR_DEF(cuDoubleComplex, Z);

// syr2
#define SYR2_DEF(type_name, short_type_name) \
	inline cublasStatus_t syr2(cublasHandle_t handle, cublasFillMode_t uplo, int n, \
			const type_name *alpha, const type_name *x, int incx, \
			const type_name *y, int incy, type_name *A, int lda){\
		return cublas##short_type_name##syr2(handle, uplo, n, alpha, x, incx, y, incy, A, lda); \
	}
SYR2_DEF(float, S);
SYR2_DEF(double, D);
SYR2_DEF(cuComplex, C);
SYR2_DEF(cuDoubleComplex, Z);

// tbmv
#define TBMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t tbmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int n, int k, const type_name *A, int lda, \
			type_name *x, int incx){\
		return cublas##short_type_name##tbmv(handle, uplo, trans, diag, n, k, A, lda, x, incx); \
	}
TBMV_DEF(float, S);
TBMV_DEF(double, D);
TBMV_DEF(cuComplex, C);
TBMV_DEF(cuDoubleComplex, Z);

// tbsv
#define TBSV_DEF(type_name, short_type_name) \
	inline cublasStatus_t tbsv(cublasHandle_t handle, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int n, int k, const type_name *A, int lda, \
			type_name *x, int incx){\
		return cublas##short_type_name##tbsv(handle, uplo, trans, diag, n, k, A, lda, x, incx); \
	}
TBSV_DEF(float, S);
TBSV_DEF(double, D);
TBSV_DEF(cuComplex, C);
TBSV_DEF(cuDoubleComplex, Z);

// tpmv
#define TPMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t tpmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int n, const type_name *AP, \
			type_name *x, int incx){\
		return cublas##short_type_name##tpmv(handle, uplo, trans, diag, n, AP, x, incx); \
	}
TPMV_DEF(float, S);
TPMV_DEF(double, D);
TPMV_DEF(cuComplex, C);
TPMV_DEF(cuDoubleComplex, Z);

// tpsv
#define TPSV_DEF(type_name, short_type_name) \
	inline cublasStatus_t tpsv(cublasHandle_t handle, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int n, const type_name *AP, \
			type_name *x, int incx){\
		return cublas##short_type_name##tpsv(handle, uplo, trans, diag, n, AP, x, incx); \
	}
TPSV_DEF(float, S);
TPSV_DEF(double, D);
TPSV_DEF(cuComplex, C);
TPSV_DEF(cuDoubleComplex, Z);

// trmv
#define TRMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t trmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int n, const type_name *A, int lda, \
			type_name *x, int incx){\
		return cublas##short_type_name##trmv(handle, uplo, trans, diag, n, A, lda, x, incx); \
	}
TRMV_DEF(float, S);
TRMV_DEF(double, D);
TRMV_DEF(cuComplex, C);
TRMV_DEF(cuDoubleComplex, Z);

// trsv
#define TRSV_DEF(type_name, short_type_name) \
	inline cublasStatus_t trsv(cublasHandle_t handle, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int n, const type_name *A, int lda, \
			type_name *x, int incx){\
		return cublas##short_type_name##trsv(handle, uplo, trans, diag, n, A, lda, x, incx); \
	}
TRSV_DEF(float, S);
TRSV_DEF(double, D);
TRSV_DEF(cuComplex, C);
TRSV_DEF(cuDoubleComplex, Z);

// hemv
#define HEMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t hemv(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *x, int incx, \
			const type_name *beta, \
			type_name *y, int incy){\
		return cublas##short_type_name##hemv(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy); \
	}
HEMV_DEF(cuComplex, C);
HEMV_DEF(cuDoubleComplex, Z);

// hbmv
#define HBMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t hbmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, int k, const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *x, int incx, \
			const type_name *beta, \
			type_name *y, int incy){\
		return cublas##short_type_name##hbmv(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy); \
	}
HBMV_DEF(cuComplex, C);
HBMV_DEF(cuDoubleComplex, Z);

// hpmv
#define HPMV_DEF(type_name, short_type_name) \
	inline cublasStatus_t hpmv(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name *alpha, \
			const type_name *AP,\
			const type_name *x, int incx, \
			const type_name *beta, \
			type_name *y, int incy){\
		return cublas##short_type_name##hpmv(handle, uplo, n, alpha, AP, x, incx, beta, y, incy); \
	}
HPMV_DEF(cuComplex, C);
HPMV_DEF(cuDoubleComplex, Z);

// her
#define HER_DEF(type_name, short_type_name, scale_type_name) \
	inline cublasStatus_t her(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const scale_type_name *alpha, \
			const type_name *x, int incx, \
			type_name *A, int lda){\
		return cublas##short_type_name##her(handle, uplo, n, alpha, x, incx, A, lda); \
	}
HER_DEF(cuComplex, C, float);
HER_DEF(cuDoubleComplex, Z, double);

// her2
#define HER2_DEF(type_name, short_type_name) \
	inline cublasStatus_t her2(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name *alpha, \
			const type_name *x, int incx, \
			const type_name *y, int incy, \
			type_name *A, int lda){\
		return cublas##short_type_name##her2(handle, uplo, n, alpha, x, incx, y, incy, A, lda); \
	}
HER2_DEF(cuComplex, C);
HER2_DEF(cuDoubleComplex, Z);

// hpr
#define HPR_DEF(type_name, short_type_name, scale_type_name) \
	inline cublasStatus_t hpr(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const scale_type_name *alpha, \
			const type_name *x, int incx, \
			type_name *AP){\
		return cublas##short_type_name##hpr(handle, uplo, n, alpha, x, incx, AP); \
	}
HPR_DEF(cuComplex, C, float);
HPR_DEF(cuDoubleComplex, Z, double);

// hpr2
#define HPR2_DEF(type_name, short_type_name) \
	inline cublasStatus_t hpr2(cublasHandle_t handle, cublasFillMode_t uplo, \
			int n, const type_name *alpha, \
			const type_name *x, int incx, \
			const type_name *y, int incy, \
			type_name *AP){\
		return cublas##short_type_name##hpr2(handle, uplo, n, alpha, x, incx, y, incy, AP); \
	}
HPR2_DEF(cuComplex, C);
HPR2_DEF(cuDoubleComplex, Z);

// ==================================================
// BLAS Lv 3
// ==================================================
// gemm
#define GEMM_DEF(type_name, short_type_name) \
	inline cublasStatus_t gemm(cublasHandle_t handle, \
			cublasOperation_t transa, cublasOperation_t transb, \
			int m, int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const type_name *beta, \
			type_name *C, int ldc) { \
		return cublas##short_type_name##gemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
GEMM_DEF(float, S);
GEMM_DEF(double, D);
GEMM_DEF(cuComplex, C);
GEMM_DEF(cuDoubleComplex, Z);
GEMM_DEF(__half, H);

// gemm3m
#define GEMM3M_DEF(type_name, short_type_name) \
	inline cublasStatus_t gemm3m(cublasHandle_t handle, \
			cublasOperation_t transa, cublasOperation_t transb, \
			int m, int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const type_name *beta, \
			type_name *C, int ldc) { \
		return cublas##short_type_name##gemm3m(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
GEMM3M_DEF(cuComplex, C);
GEMM3M_DEF(cuDoubleComplex, Z);

// gemmBatched
#define GEMM_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t gemm_batched(cublasHandle_t handle, \
			cublasOperation_t transa,  \
			cublasOperation_t transb, \
			int m, int n, int k, \
			const type_name *alpha, \
			const type_name *Aarray[], int lda, \
			const type_name *Barray[], int ldb, \
			const type_name *beta, \
			type_name *Carray[], int ldc,  \
			int batchCount) { \
		return cublas##short_type_name##gemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount); \
	}
GEMM_BATCHED_DEF(float, S);
GEMM_BATCHED_DEF(double, D);
GEMM_BATCHED_DEF(cuComplex, C);
GEMM_BATCHED_DEF(cuDoubleComplex, Z);
GEMM_BATCHED_DEF(__half, H);

// gemmStridedBatched
#define GEMM_STRIDED_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t gemm_strided_batched(cublasHandle_t handle, \
			cublasOperation_t transa,  \
			cublasOperation_t transb, \
			int m, int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			long long int strideA, \
			const type_name *B, int ldb, \
			long long int strideB, \
			const type_name *beta, \
			type_name *C, int ldc,  \
			long long int strideC, \
			int batchCount){ \
		return cublas##short_type_name##gemmStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount); \
	}
GEMM_STRIDED_BATCHED_DEF(float, S);
GEMM_STRIDED_BATCHED_DEF(double, D);
GEMM_STRIDED_BATCHED_DEF(cuComplex, C);
GEMM_STRIDED_BATCHED_DEF(cuDoubleComplex, Z);
GEMM_STRIDED_BATCHED_DEF(__half, H);

// gemm3mStridedBatched
#define GEMM3M_STRIDED_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t gemm3m_strided_batched(cublasHandle_t handle, \
			cublasOperation_t transa,  \
			cublasOperation_t transb, \
			int m, int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			long long int strideA, \
			const type_name *B, int ldb, \
			long long int strideB, \
			const type_name *beta, \
			type_name *C, int ldc,  \
			long long int strideC, \
			int batchCount){ \
		return cublas##short_type_name##gemm3mStridedBatched(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount); \
	}
GEMM3M_STRIDED_BATCHED_DEF(cuComplex, C);

// symm
#define SYMM_DEF(type_name, short_type_name) \
	inline cublasStatus_t symm(cublasHandle_t handle, \
			cublasSideMode_t side, cublasFillMode_t uplo, \
			int m, int n, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const type_name *beta, \
			type_name *C, int ldc) { \
		return cublas##short_type_name##symm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
SYMM_DEF(float, S);
SYMM_DEF(double, D);
SYMM_DEF(cuComplex, C);
SYMM_DEF(cuDoubleComplex, Z);

// syrk
#define SYRK_DEF(type_name, short_type_name) \
	inline cublasStatus_t syrk(cublasHandle_t handle, \
			cublasFillMode_t uplo, cublasOperation_t trans, \
			int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *beta, \
			type_name *C, int ldc) { \
		return cublas##short_type_name##syrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc); \
	}
SYRK_DEF(float, S);
SYRK_DEF(double, D);
SYRK_DEF(cuComplex, C);
SYRK_DEF(cuDoubleComplex, Z);

// syr2k
#define SYR2K_DEF(type_name, short_type_name) \
	inline cublasStatus_t syr2k(cublasHandle_t handle, \
			cublasFillMode_t uplo, cublasOperation_t trans, \
			int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const type_name *beta, \
			type_name *C, int ldc) { \
		return cublas##short_type_name##syr2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
SYR2K_DEF(float, S);
SYR2K_DEF(double, D);
SYR2K_DEF(cuComplex, C);
SYR2K_DEF(cuDoubleComplex, Z);

// syrkx
#define SYRKX_DEF(type_name, short_type_name) \
	inline cublasStatus_t syrkx(cublasHandle_t handle, \
			cublasFillMode_t uplo, cublasOperation_t trans, \
			int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const type_name *beta, \
			type_name *C, int ldc){ \
		return cublas##short_type_name##syrkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
SYRKX_DEF(float, S);
SYRKX_DEF(double, D);
SYRKX_DEF(cuComplex, C);
SYRKX_DEF(cuDoubleComplex, Z);

// trmm
#define TRMM_DEF(type_name, short_type_name) \
	inline cublasStatus_t trmm(cublasHandle_t handle, \
			cublasSideMode_t side, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int m, int n, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			type_name *C, int ldc){ \
		return cublas##short_type_name##trmm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, C, ldc); \
	}
TRMM_DEF(float, S);
TRMM_DEF(double, D);
TRMM_DEF(cuComplex, C);
TRMM_DEF(cuDoubleComplex, Z);

// trsm
#define TRSM_DEF(type_name, short_type_name) \
	inline cublasStatus_t trsm(cublasHandle_t handle, \
			cublasSideMode_t side, cublasFillMode_t uplo, \
			cublasOperation_t trans, cublasDiagType_t diag, \
			int m, int n, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			type_name *B, int ldb){ \
		return cublas##short_type_name##trsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb); \
	}
TRSM_DEF(float, S);
TRSM_DEF(double, D);
TRSM_DEF(cuComplex, C);
TRSM_DEF(cuDoubleComplex, Z);

// trsmBatched
#define TRSM_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t trsm_batched( cublasHandle_t handle,  \
			cublasSideMode_t side,  \
			cublasFillMode_t uplo, \
			cublasOperation_t trans,  \
			cublasDiagType_t diag, \
			int m,  \
			int n,  \
			const type_name *alpha, \
			type_name *A[],  \
			int lda, \
			type_name *B[],  \
			int ldb, \
			int batchCount){ \
		return cublas##short_type_name##trsmBatched(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batchCount); \
	}
TRSM_BATCHED_DEF(float, S);
TRSM_BATCHED_DEF(double, D);
TRSM_BATCHED_DEF(cuComplex, C);
TRSM_BATCHED_DEF(cuDoubleComplex, Z);

// hemm
#define HEMM_DEF(type_name, short_type_name) \
	inline cublasStatus_t hemm(cublasHandle_t handle, \
			cublasSideMode_t side, cublasFillMode_t uplo, \
			int m, int n, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const type_name *beta, \
			type_name *C, int ldc){ \
		return cublas##short_type_name##hemm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
HEMM_DEF(cuComplex, C);
HEMM_DEF(cuDoubleComplex, Z);

// herk
#define HERK_DEF(type_name, short_type_name, ab_type_name) \
	inline cublasStatus_t herk(cublasHandle_t handle, \
			cublasFillMode_t uplo, cublasOperation_t trans, \
			int n, int k, \
			const ab_type_name *alpha, \
			const type_name *A, int lda, \
			const ab_type_name *beta, \
			type_name *C, int ldc){ \
		return cublas##short_type_name##herk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc); \
	}
HERK_DEF(cuComplex, C, float);
HERK_DEF(cuDoubleComplex, Z, double);

// her2k
#define HER2K_DEF(type_name, short_type_name, ab_type_name) \
	inline cublasStatus_t her2k(cublasHandle_t handle, \
			cublasFillMode_t uplo, cublasOperation_t trans, \
			int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const ab_type_name *beta, \
			type_name *C, int ldc) { \
		return cublas##short_type_name##her2k(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
HER2K_DEF(cuComplex, C, float);
HER2K_DEF(cuDoubleComplex, Z, double);

// her2k
#define HERKX_DEF(type_name, short_type_name, ab_type_name) \
	inline cublasStatus_t herkx(cublasHandle_t handle, \
			cublasFillMode_t uplo, cublasOperation_t trans, \
			int n, int k, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *B, int ldb, \
			const ab_type_name  *beta, \
			type_name       *C, int ldc){ \
		return cublas##short_type_name##herkx(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc); \
	}
HERKX_DEF(cuComplex, C, float);
HERKX_DEF(cuDoubleComplex, Z, double);

// ==================================================
// BLAS Lv 1
// ==================================================
// geam
#define GEAM_DEF(type_name, short_type_name) \
	inline cublasStatus_t geam(cublasHandle_t handle, \
			cublasOperation_t transa, cublasOperation_t transb, \
			int m, int n, \
			const type_name *alpha, \
			const type_name *A, int lda, \
			const type_name *beta, \
			const type_name *B, int ldb, \
			type_name *C, int ldc){ \
		return cublas##short_type_name##geam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc); \
	}
GEAM_DEF(float, S);
GEAM_DEF(double, D);
GEAM_DEF(cuComplex, C);
GEAM_DEF(cuDoubleComplex, Z);

// dgmm
#define DGMM_DEF(type_name, short_type_name) \
	inline cublasStatus_t dgmm(cublasHandle_t handle, cublasSideMode_t mode, \
			int m, int n, \
			const type_name *A, int lda, \
			const type_name *x, int incx, \
			type_name *C, int ldc){ \
		return cublas##short_type_name##dgmm(handle, mode, m, n, A, lda, x, incx, C, ldc); \
	}
DGMM_DEF(float, S);
DGMM_DEF(double, D);
DGMM_DEF(cuComplex, C);
DGMM_DEF(cuDoubleComplex, Z);

// getrfBatched
#define GETRF_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t getrf_batched(cublasHandle_t handle, \
			int n,  \
			type_name *Aarray[], \
			int lda,  \
			int *PivotArray, \
			int *infoArray, \
			int batchSize){ \
		return cublas##short_type_name##getrfBatched(handle, n, Aarray, lda, PivotArray, infoArray, batchSize); \
	}
GETRF_BATCHED_DEF(float, S);
GETRF_BATCHED_DEF(double, D);
GETRF_BATCHED_DEF(cuComplex, C);
GETRF_BATCHED_DEF(cuDoubleComplex, Z);

// getrsBatched
#define GETRS_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t getrs_batched(cublasHandle_t handle, \
			cublasOperation_t trans,  \
			int n,  \
			int nrhs,  \
			const type_name *Aarray[],  \
			int lda,  \
			const int *devIpiv,  \
			type_name *Barray[],  \
			int ldb,  \
			int *info, \
			int batchSize){ \
		return cublas##short_type_name##getrsBatched(handle, trans, n, nrhs, Aarray, lda, devIpiv, Barray, ldb, info, batchSize); \
	}
GETRS_BATCHED_DEF(float, S);
GETRS_BATCHED_DEF(double, D);
GETRS_BATCHED_DEF(cuComplex, C);
GETRS_BATCHED_DEF(cuDoubleComplex, Z);

// getriBatched
#define GETRI_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t getri_batched(cublasHandle_t handle, \
			int n, \
			type_name *Aarray[], \
			int lda, \
			int *PivotArray, \
			type_name *Carray[], \
			int ldc, \
			int *infoArray, \
			int batchSize){ \
		return cublas##short_type_name##getriBatched(handle, n, Aarray, lda, PivotArray, Carray, ldc, infoArray, batchSize); \
	}
GETRI_BATCHED_DEF(float, S);
GETRI_BATCHED_DEF(double, D);
GETRI_BATCHED_DEF(cuComplex, C);
GETRI_BATCHED_DEF(cuDoubleComplex, Z);

// matinvBatched
#define MATINV_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t matinv_batched(cublasHandle_t handle, \
			int n,  \
			const type_name *A[], \
			int lda, \
			type_name *Ainv[], \
			int lda_inv, \
			int *info, \
			int batchSize){ \
		return cublas##short_type_name##matinvBatched(handle, n, A, lda, Ainv, lda_inv, info, batchSize); \
	}
MATINV_BATCHED_DEF(float, S);
MATINV_BATCHED_DEF(double, D);
MATINV_BATCHED_DEF(cuComplex, C);
MATINV_BATCHED_DEF(cuDoubleComplex, Z);

// geqrfBatched
#define GEQRF_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t geqrf_batched( cublasHandle_t handle,  \
			int m,  \
			int n, \
			type_name *Aarray[],   \
			int lda,  \
			type_name *TauArray[],  \
			int *info, \
			int batchSize){ \
		return cublas##short_type_name##geqrfBatched(handle, m, n, Aarray, lda, TauArray, info, batchSize); \
	}
GEQRF_BATCHED_DEF(float, S);
GEQRF_BATCHED_DEF(double, D);
GEQRF_BATCHED_DEF(cuComplex, C);
GEQRF_BATCHED_DEF(cuDoubleComplex, Z);

// gelsBatched
#define GELS_BATCHED_DEF(type_name, short_type_name) \
	inline cublasStatus_t gels_batched( cublasHandle_t handle, \
			cublasOperation_t trans, \
			int m, \
			int n, \
			int nrhs, \
			type_name *Aarray[], \
			int lda, \
			type_name *Carray[], \
			int ldc, \
			int *info, \
			int *devInfoArray, \
			int batchSize ){ \
		return cublas##short_type_name##gelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize); \
	}
GELS_BATCHED_DEF(float, S);
GELS_BATCHED_DEF(double, D);
GELS_BATCHED_DEF(cuComplex, C);
GELS_BATCHED_DEF(cuDoubleComplex, Z);

// tpttr
#define TPTTR_DEF(type_name, short_type_name) \
	inline cublasStatus_t tpttr( cublasHandle_t handle, \
			cublasFillMode_t uplo, \
			int n, \
			const type_name *AP, \
			type_name *A, \
			int lda ) { \
		return cublas##short_type_name##tpttr(handle, uplo, n, AP, A, lda); \
	}
TPTTR_DEF(float, S);
TPTTR_DEF(double, D);
TPTTR_DEF(cuComplex, C);
TPTTR_DEF(cuDoubleComplex, Z);

// trttp
#define TRTTP_DEF(type_name, short_type_name) \
	inline cublasStatus_t trttp( cublasHandle_t handle, \
			cublasFillMode_t uplo, \
			int n, \
			const type_name *A, \
			int lda, \
			type_name *AP ){ \
		return cublas##short_type_name##trttp(handle, uplo, n, A, lda, AP); \
	}
TRTTP_DEF(float, S);
TRTTP_DEF(double, D);
TRTTP_DEF(cuComplex, C);
TRTTP_DEF(cuDoubleComplex, Z);

} // cublas
} // cutf

#endif // __CUTF_CUBLAS_CUH__
