#ifndef __CUTF_CUSOLVER_HPP__
#define __CUTF_CUSOLVER_HPP__
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <sstream>
#include <memory>
#include "cuda.hpp"

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
inline std::unique_ptr<cusolverSpHandle_t, cusolver_sp_deleter> get_cusolver_sp_unique_ptr(const int device_id = 0){
	cutf::error::check(cudaSetDevice(device_id), __FILE__, __LINE__, __func__);
	cusolverSpHandle_t *handle = new cusolverSpHandle_t;
	cusolverSpCreate(handle);
	cutf::error::check(cudaSetDevice(0), __FILE__, __LINE__, __func__);
	return std::unique_ptr<cusolverSpHandle_t, cusolver_sp_deleter>{handle};
}
struct cusolver_dn_deleter{
	void operator()(cusolverDnHandle_t* handle){
		cutf::error::check(cusolverDnDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
inline std::unique_ptr<cusolverDnHandle_t, cusolver_dn_deleter> get_cusolver_dn_unique_ptr(const int device_id = 0){
	cutf::error::check(cudaSetDevice(device_id), __FILE__, __LINE__, __func__);
	cusolverDnHandle_t *handle = new cusolverDnHandle_t;
	cusolverDnCreate(handle);
	cutf::error::check(cudaSetDevice(0), __FILE__, __LINE__, __func__);
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
} // namespace dn

} // cusolver
} // cutf

#endif // __CUTF_CUSOLVER_HPP__