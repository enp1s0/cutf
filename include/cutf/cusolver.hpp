#ifndef __CUTF_CUSOLVER_HPP__
#define __CUTF_CUSOLVER_HPP__
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <sstream>
#include <memory>
#include "cuda.hpp"
#include "error.hpp"

namespace cutf{
namespace error{
inline void check(cusolverStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUSOLVER_STATUS_SUCCESS){
		std::string error_string;
#define CUSOLVER_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_NOT_INITIALIZED );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_ALLOC_FAILED );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_INVALID_VALUE );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_ARCH_MISMATCH );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_EXECUTION_FAILED );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_INTERNAL_ERROR );
			CUSOLVER_ERROR_CASE( CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED );
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
namespace cusolver{
namespace sp {
struct handle_deleter{
	void operator()(cusolverSpHandle_t* handle){
		cutf::error::check(cusolverSpDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
using handle_unique_ptr = std::unique_ptr<cusolverSpHandle_t, handle_deleter>;

inline handle_unique_ptr get_handle_unique_ptr(){
	auto *handle = new cusolverSpHandle_t;
	cusolverSpCreate(handle);
	return handle_unique_ptr{handle};
}
} // namespace sp

namespace dn {
struct handle_deleter{
	void operator()(cusolverDnHandle_t* handle){
		cutf::error::check(cusolverDnDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
using handle_unique_ptr = std::unique_ptr<cusolverDnHandle_t, handle_deleter>;

inline handle_unique_ptr get_handle_unique_ptr(){
	auto *handle = new cusolverDnHandle_t;
	cusolverDnCreate(handle);
	return handle_unique_ptr{handle};
}

struct params_deleter{
	void operator()(cusolverDnParams_t* params){
		cutf::error::check(cusolverDnDestroyParams(*params), __FILE__, __LINE__, __func__);
		delete params;
	}
};
using params_unique_ptr = std::unique_ptr<cusolverDnParams_t, params_deleter>;

inline params_unique_ptr get_params_unique_ptr(){
	auto *params = new cusolverDnParams_t;
	cusolverDnCreateParams(params);
	return params_unique_ptr{params};
}

// --------------------------------------------------------------------------
// Choresky Factorization
// --------------------------------------------------------------------------
#define POTRF(type_name, short_type_name)\
	inline cusolverStatus_t potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, type_name *A, int lda, type_name *Workspace, int Lwork, int *devInfo) {\
		return cusolverDn##short_type_name##potrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo); \
	} \
	inline cusolverStatus_t potrf_buffer_size(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, type_name *A, int lda, int *Lwork) {\
		return cusolverDn##short_type_name##potrf_bufferSize(handle, uplo, n, A, lda, Lwork);\
	}
POTRF(float, S);
POTRF(double, D);
POTRF(cuComplex, C);
POTRF(cuDoubleComplex, Z);

#define POTRS(type_name, short_type_name)\
	inline cusolverStatus_t potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const type_name *A, int lda, type_name *B, int ldb, int *devInfo) {\
		return cusolverDn##short_type_name##potrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo); \
	}
POTRS(float, S);
POTRS(double, D);
POTRS(cuComplex, C);
POTRS(cuDoubleComplex, Z);

#define POTRI(type_name, short_type_name)\
	inline cusolverStatus_t potri(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, type_name *A, int lda, type_name *Workspace, int Lwork, int *devInfo) {\
		return cusolverDn##short_type_name##potri(handle, uplo, n, A, lda, Workspace, Lwork, devInfo); \
	} \
	inline cusolverStatus_t potri_buffer_size(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, type_name *A, int lda, int *Lwork) {\
		return cusolverDn##short_type_name##potri_bufferSize(handle, uplo, n, A, lda, Lwork);\
	}
POTRI(float, S);
POTRI(double, D);
POTRI(cuComplex, C);
POTRI(cuDoubleComplex, Z);

// --------------------------------------------------------------------------
// LU Factorization
// --------------------------------------------------------------------------
#define GETRF(type_name, short_type_name)\
	inline cusolverStatus_t getrf(cusolverDnHandle_t handle, int m, int n, type_name *A, int lda, type_name *Workspace, int *devIpiv, int *devInfo) {\
		return cusolverDn##short_type_name##getrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo); \
	} \
	inline cusolverStatus_t getrf_buffer_size(cusolverDnHandle_t handle, int m, int n, type_name *A, int lda, int *Lwork) {\
		return cusolverDn##short_type_name##getrf_bufferSize(handle, m, n, A, lda, Lwork);\
	}
GETRF(float, S);
GETRF(double, D);
GETRF(cuComplex, C);
GETRF(cuDoubleComplex, Z);

#define GETRS(type_name, short_type_name)\
	inline cusolverStatus_t getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const type_name *A, int lda, const int *devIpiv, type_name *B, int ldb, int *devInfo) {\
		return cusolverDn##short_type_name##getrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo); \
	}
GETRS(float, S);
GETRS(double, D);
GETRS(cuComplex, C);
GETRS(cuDoubleComplex, Z);


// --------------------------------------------------------------------------
// QR Factorization
// --------------------------------------------------------------------------
#define GEQRF(type_name, short_type_name)\
	inline cusolverStatus_t geqrf(cusolverDnHandle_t handle, int m, int n, type_name *A, int lda, type_name* TAU, type_name *Workspace, int Lwork, int *devInfo) {\
		return cusolverDn##short_type_name##geqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo); \
	} \
	inline cusolverStatus_t geqrf_buffer_size(cusolverDnHandle_t handle, int m, int n, type_name *A, int lda, int *Lwork) {\
		return cusolverDn##short_type_name##geqrf_bufferSize(handle, m, n, A, lda, Lwork);\
	}
GEQRF(float, S);
GEQRF(double, D);
GEQRF(cuComplex, C);
GEQRF(cuDoubleComplex, Z);

#define MQR(type_name, short_type_name, matrix_name)\
	inline cusolverStatus_t matrix_name##mqr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, type_name *A, int lda, const type_name* tau, type_name* C, int ldc, type_name *work, int lwork, int *devInfo) {\
		return cusolverDn##short_type_name##matrix_name##mqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t matrix_name##mqr_buffer_size(cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, const type_name *A, int lda, const type_name* tau, const type_name *C, int ldc, int *lwork) {\
		return cusolverDn##short_type_name##matrix_name##mqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);\
	}
MQR(float, S, or);
MQR(double, D, or);
MQR(cuComplex, C, un);
MQR(cuDoubleComplex, Z, un);

#define GQR(type_name, short_type_name, matrix_name)\
	inline cusolverStatus_t matrix_name##gqr(cusolverDnHandle_t handle, int m, int n, int k, type_name *A, int lda, const type_name* tau, type_name *work, int lwork, int *devInfo) {\
		return cusolverDn##short_type_name##matrix_name##gqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t matrix_name##gqr_buffer_size(cusolverDnHandle_t handle, int m, int n, int k, const type_name *A, int lda, const type_name* tau, int *lwork) {\
		return cusolverDn##short_type_name##matrix_name##gqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);\
	}
GQR(float, S, or);
GQR(double, D, or);
GQR(cuComplex, C, un);
GQR(cuDoubleComplex, Z, un);

// --------------------------------------------------------------------------
// Bunch-Kaufman Factorization
// --------------------------------------------------------------------------
#define SYTRF(type_name, short_type_name)\
	inline cusolverStatus_t sytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, type_name *A, int lda, int *ipiv, type_name *work, int lwork, int *devInfo) {\
		return cusolverDn##short_type_name##sytrf(handle, uplo, n, A, lda, ipiv, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t sytrf_buffer_size(cusolverDnHandle_t handle, int n, type_name *A, int lda, int *Lwork) {\
		return cusolverDn##short_type_name##sytrf_bufferSize(handle, n, A, lda, Lwork);\
	}
SYTRF(float, S);
SYTRF(double, D);
SYTRF(cuComplex, C);
SYTRF(cuDoubleComplex, Z);

// --------------------------------------------------------------------------
// Q^H * A * P = B
// --------------------------------------------------------------------------
template <class T>
inline cusolverStatus_t gebrd_buffer_size(cusolverDnHandle_t handle, const int m, const int n, int *Lwork);
#define GEBRD(type_name, DE_type_name, short_type_name)\
	inline cusolverStatus_t gebrd(cusolverDnHandle_t handle, const int m, const int n, type_name *A, const int lda, DE_type_name* D, DE_type_name* E, type_name* TAUQ, type_name* TAUP, type_name* Work, const int Lwork, int* devInfo) {\
		return cusolverDn##short_type_name##gebrd(handle, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo); \
	} \
  template <> \
	inline cusolverStatus_t gebrd_buffer_size<type_name>(cusolverDnHandle_t handle, const int m, const int n, int *Lwork) {\
		return cusolverDn##short_type_name##gebrd_bufferSize(handle, m, n, Lwork);\
	}
GEBRD(float, float, S);
GEBRD(double, double, D);
GEBRD(cuComplex, float, C);
GEBRD(cuDoubleComplex, double, Z);

#define GBR(type_name, matrix_name, short_type_name)\
	inline cusolverStatus_t matrix_name##gbr(cusolverDnHandle_t handle, cublasSideMode_t side, const int m, const int n, const int k, type_name *A, const int lda, type_name* tau, type_name* work, const int lwork, int* devInfo) {\
		return cusolverDn##short_type_name##matrix_name##gbr(handle, side, m, n, k, A, lda, tau, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t matrix_name##gbr_buffer_size(cusolverDnHandle_t handle, cublasSideMode_t side, const int m, const int n, const int k, const type_name* A, const int lda, const type_name* tau, int* lwork) {\
		return cusolverDn##short_type_name##matrix_name##gbr_bufferSize(handle, side, m, n, k, A, lda, tau, lwork);\
	}
GBR(float, or, S);
GBR(double, or, D);
GBR(cuComplex, un, C);
GBR(cuDoubleComplex, un, Z);

// --------------------------------------------------------------------------
// Q^H * A * Q = T
// --------------------------------------------------------------------------
#define TRD(type_name, DE_type_name, matrix_name, short_type_name)\
	inline cusolverStatus_t matrix_name##trd(cusolverDnHandle_t handle, cublasFillMode_t uplo, const int n, type_name *A, const int lda, DE_type_name* d, DE_type_name* e, type_name* tau, type_name* work, const int lwork, int* devInfo) {\
		return cusolverDn##short_type_name##matrix_name##trd(handle, uplo, n, A, lda, d, e, tau, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t matrix_name##trd_buffer_size(cusolverDnHandle_t handle, cublasFillMode_t uplo, const int n, const type_name* A, const int lda, const DE_type_name* d, const DE_type_name* e, const type_name* tau, int *lwork) {\
		return cusolverDn##short_type_name##matrix_name##trd_bufferSize(handle, uplo, n, A, lda, d, e, tau, lwork);\
	}
TRD(float, float, sy, S);
TRD(double, double, sy, D);
TRD(cuComplex, float, he, C);
TRD(cuDoubleComplex, double, he, Z);

// --------------------------------------------------------------------------
// multiply Q
// --------------------------------------------------------------------------
#define MTR(type_name, matrix_name, short_type_name)\
	inline cusolverStatus_t matrix_name##mtr(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, const int m, const int n, type_name *A, const int lda, type_name* tau, type_name* C, const int ldc, type_name* work, const int lwork, int* devInfo) {\
		return cusolverDn##short_type_name##matrix_name##mtr(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t matrix_name##trm_buffer_size(cusolverDnHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, const int m, const int n, type_name *A, const int lda, type_name* tau, type_name* C, const int ldc, int *lwork) {\
		return cusolverDn##short_type_name##matrix_name##mtr_bufferSize(handle, side, uplo, trans, m, n, A, lda, tau, C, ldc, lwork);\
	}
MTR(float, or, S);
MTR(double, or, D);
MTR(cuComplex, un, C);
MTR(cuDoubleComplex, un, Z);

// --------------------------------------------------------------------------
// Generate Q
// --------------------------------------------------------------------------
#define GTR(type_name, matrix_name, short_type_name)\
	inline cusolverStatus_t matrix_name##gtr(cusolverDnHandle_t handle, cublasFillMode_t uplo, const int n, type_name *A, const int lda, type_name* tau, type_name* work, const int lwork, int* devInfo) {\
		return cusolverDn##short_type_name##matrix_name##gtr(handle, uplo, n, A, lda, tau, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t matrix_name##gtr_buffer_size(cusolverDnHandle_t handle, cublasFillMode_t uplo, const int n, type_name *A, const int lda, type_name* tau, int *lwork) {\
		return cusolverDn##short_type_name##matrix_name##gtr_bufferSize(handle, uplo, n, A, lda, tau, lwork);\
	}
GTR(float, or, S);
GTR(double, or, D);
GTR(cuComplex, un, C);
GTR(cuDoubleComplex, un, Z);

// --------------------------------------------------------------------------
// SVD
// --------------------------------------------------------------------------
template <class T>
inline cusolverStatus_t gesvd_buffer_size(cusolverDnHandle_t handle, int m, int n, int *Lwork);
#define GESVD(type_name, s_type_name, short_type_name)\
	inline cusolverStatus_t gesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n, type_name* A, int lda, s_type_name* S, type_name* U, int ldu, type_name* VT, int lvt, type_name* work, int lwork, s_type_name* rwork, int *devInfo) {\
		return cusolverDn##short_type_name##gesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, lvt, work, lwork, rwork, devInfo); \
	} \
	template <> inline cusolverStatus_t gesvd_buffer_size<type_name>(cusolverDnHandle_t handle, int m, int n, int *Lwork) {\
		return cusolverDn##short_type_name##gesvd_bufferSize(handle, m, n, Lwork);\
	}
GESVD(float, float, S);
GESVD(double, double, D);
GESVD(cuComplex, float, C);
GESVD(cuDoubleComplex, double, Z);

#define GESVDJ(type_name, s_type_name, short_type_name)\
	inline cusolverStatus_t gesvdj(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, type_name* A, int lda, s_type_name* S, type_name* U, int ldu, type_name* V, int ldv, type_name* work, int lwork, int *info, gesvdjInfo_t params) {\
		return cusolverDn##short_type_name##gesvdj(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params); \
	} \
	inline cusolverStatus_t gesvdj_buffer_size(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, type_name* A, int lda, s_type_name* S, type_name* U, int ldu, type_name* V, int ldv, int *lwork, gesvdjInfo_t params) {\
		return cusolverDn##short_type_name##gesvdj_bufferSize(handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params); \
	}
GESVDJ(float, float, S);
GESVDJ(double, double, D);
GESVDJ(cuComplex, float, C);
GESVDJ(cuDoubleComplex, double, Z);
} // namespace dn

} // cusolver
} // cutf

#endif // __CUTF_CUSOLVER_HPP__
