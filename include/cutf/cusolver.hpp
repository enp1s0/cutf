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
struct cusolver_sp_deleter{
	void operator()(cusolverSpHandle_t* handle){
		cutf::error::check(cusolverSpDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
inline std::unique_ptr<cusolverSpHandle_t, cusolver_sp_deleter> get_cusolver_sp_unique_ptr(){
	cusolverSpHandle_t *handle = new cusolverSpHandle_t;
	cusolverSpCreate(handle);
	return std::unique_ptr<cusolverSpHandle_t, cusolver_sp_deleter>{handle};
}
struct cusolver_dn_deleter{
	void operator()(cusolverDnHandle_t* handle){
		cutf::error::check(cusolverDnDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
inline std::unique_ptr<cusolverDnHandle_t, cusolver_dn_deleter> get_cusolver_dn_unique_ptr(){
	cusolverDnHandle_t *handle = new cusolverDnHandle_t;
	cusolverDnCreate(handle);
	return std::unique_ptr<cusolverDnHandle_t, cusolver_dn_deleter>{handle};
}

namespace dn {
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

#define GQR(type_name, short_type_name)\
	inline cusolverStatus_t gqr(cusolverDnHandle_t handle, int m, int n, int k, type_name *A, int lda, const type_name* tau, type_name *work, int lwork, int *devInfo) {\
		return cusolverDn##short_type_name##gqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo); \
	} \
	inline cusolverStatus_t gqr_buffer_size(cusolverDnHandle_t handle, int m, int n, int k, const type_name *A, int lda, const type_name* tau, int *lwork) {\
		return cusolverDn##short_type_name##gqr_bufferSize(handle, m, n, k, A, lda, tau, lwork);\
	}
GQR(float, Sor);
GQR(double, Dor);
GQR(cuComplex, Cun);
GQR(cuDoubleComplex, Zun);

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
