#ifndef __CUTF_ERROR_CUH__
#define __CUTF_ERROR_CUH__
#include <stdexcept>
#include <sstream>
#include <cublas_v2.h>

namespace cutf{
namespace cuda{
namespace error{
inline void check(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss<< cudaGetErrorString( error ) <<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}

} // error
} // cuda
namespace cublas{
namespace error{
inline void check(cublasStatus_t error, const std::string filename, const std::size_t line, const std::string funcname){
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
		ss<< error_string <<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}

} // error
} // cuda
} // cutf

#endif // __CUTF_ERROR_CUH__
