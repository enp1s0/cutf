#ifndef __CUTF_ERROR_CUH__
#define __CUTF_ERROR_CUH__
#include <stdexcept>
#include <sstream>
#include <cuda_device_runtime_api.h>
#include <cuda.h>
#include "error.hpp"

namespace cutf{
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
} // cutf

#endif // __CUTF_ERROR_CUH__
