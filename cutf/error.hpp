#ifndef __CUTF_ERROR_CUH__
#define __CUTF_ERROR_CUH__
#include <stdexcept>
#include <sstream>
#include <cublas_v2.h>
#include <cuda.h>

namespace cutf{
namespace cuda{
namespace error{
inline void check(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss<< cudaGetErrorString( error );
		if(message.length() != 0){
			ss<<" : "<<message;
		}
	    ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}

} // error
} // cuda
namespace cublas{
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
} // cublas
namespace driver{
namespace error{
inline void check(CUresult error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUDA_SUCCESS){
		std::string error_string;
#define CU_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
			CU_ERROR_CASE( CUDA_ERROR_INVALID_VALUE                  );
			CU_ERROR_CASE( CUDA_ERROR_OUT_OF_MEMORY                  );
			CU_ERROR_CASE( CUDA_ERROR_NOT_INITIALIZED                );
			CU_ERROR_CASE( CUDA_ERROR_DEINITIALIZED                  );
			CU_ERROR_CASE( CUDA_ERROR_PROFILER_DISABLED              );
			CU_ERROR_CASE( CUDA_ERROR_PROFILER_NOT_INITIALIZED       );
			CU_ERROR_CASE( CUDA_ERROR_PROFILER_ALREADY_STARTED       );
			CU_ERROR_CASE( CUDA_ERROR_PROFILER_ALREADY_STOPPED       );
			CU_ERROR_CASE( CUDA_ERROR_NO_DEVICE                      );
			CU_ERROR_CASE( CUDA_ERROR_INVALID_DEVICE                 );
			CU_ERROR_CASE( CUDA_ERROR_INVALID_IMAGE                  );
			CU_ERROR_CASE( CUDA_ERROR_INVALID_CONTEXT                );
			CU_ERROR_CASE( CUDA_ERROR_CONTEXT_ALREADY_CURRENT        );
			CU_ERROR_CASE( CUDA_ERROR_MAP_FAILED                     );
			CU_ERROR_CASE( CUDA_ERROR_UNMAP_FAILED                   );
			CU_ERROR_CASE( CUDA_ERROR_ARRAY_IS_MAPPED                );
			CU_ERROR_CASE( CUDA_ERROR_ALREADY_MAPPED                 );
			CU_ERROR_CASE( CUDA_ERROR_NO_BINARY_FOR_GPU              );
			CU_ERROR_CASE( CUDA_ERROR_ALREADY_ACQUIRED               );
			CU_ERROR_CASE( CUDA_ERROR_NOT_MAPPED                     );
			CU_ERROR_CASE( CUDA_ERROR_NOT_MAPPED_AS_ARRAY            );
			CU_ERROR_CASE( CUDA_ERROR_NOT_MAPPED_AS_POINTER          );
			CU_ERROR_CASE( CUDA_ERROR_ECC_UNCORRECTABLE              );
			CU_ERROR_CASE( CUDA_ERROR_UNSUPPORTED_LIMIT              );
			CU_ERROR_CASE( CUDA_ERROR_CONTEXT_ALREADY_IN_USE         );
			CU_ERROR_CASE( CUDA_ERROR_INVALID_SOURCE                 );
			CU_ERROR_CASE( CUDA_ERROR_FILE_NOT_FOUND                 );
			CU_ERROR_CASE( CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND );
			CU_ERROR_CASE( CUDA_ERROR_SHARED_OBJECT_INIT_FAILED      );
			CU_ERROR_CASE( CUDA_ERROR_OPERATING_SYSTEM               );
			CU_ERROR_CASE( CUDA_ERROR_INVALID_HANDLE                 );
			CU_ERROR_CASE( CUDA_ERROR_NOT_FOUND                      );
			CU_ERROR_CASE( CUDA_ERROR_NOT_READY                      );
			CU_ERROR_CASE( CUDA_ERROR_LAUNCH_FAILED                  );
			CU_ERROR_CASE( CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES        );
			CU_ERROR_CASE( CUDA_ERROR_LAUNCH_TIMEOUT                 );
			CU_ERROR_CASE( CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING  );
			CU_ERROR_CASE( CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED    );
			CU_ERROR_CASE( CUDA_ERROR_PEER_ACCESS_NOT_ENABLED        );
			CU_ERROR_CASE( CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE         );
			CU_ERROR_CASE( CUDA_ERROR_CONTEXT_IS_DESTROYED           );
			CU_ERROR_CASE( CUDA_ERROR_ASSERT                         );
			CU_ERROR_CASE( CUDA_ERROR_TOO_MANY_PEERS                 );
			CU_ERROR_CASE( CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED );
			CU_ERROR_CASE( CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED     );
			CU_ERROR_CASE( CUDA_ERROR_UNKNOWN                        );
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
} // cublas
} // cutf

#endif // __CUTF_ERROR_CUH__
