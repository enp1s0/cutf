#ifndef __CUTF_NVML_HPP__
#define __CUTF_NVML_HPP__
#include <memory>
#include <sstream>
#include <nvml.h>
#include "error.hpp"

namespace cutf{
namespace error{
inline void check(nvmlReturn_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != NVML_SUCCESS){
		std::string error_string;
#define NVML_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
			NVML_ERROR_CASE(NVML_ERROR_UNINITIALIZED);
			NVML_ERROR_CASE(NVML_ERROR_INVALID_ARGUMENT);
			NVML_ERROR_CASE(NVML_ERROR_NOT_SUPPORTED);
			NVML_ERROR_CASE(NVML_ERROR_NO_PERMISSION);
			NVML_ERROR_CASE(NVML_ERROR_ALREADY_INITIALIZED);
			NVML_ERROR_CASE(NVML_ERROR_NOT_FOUND);
			NVML_ERROR_CASE(NVML_ERROR_INSUFFICIENT_SIZE);
			NVML_ERROR_CASE(NVML_ERROR_INSUFFICIENT_POWER);
			NVML_ERROR_CASE(NVML_ERROR_DRIVER_NOT_LOADED);
			NVML_ERROR_CASE(NVML_ERROR_TIMEOUT);
			NVML_ERROR_CASE(NVML_ERROR_IRQ_ISSUE);
			NVML_ERROR_CASE(NVML_ERROR_LIBRARY_NOT_FOUND);
			NVML_ERROR_CASE(NVML_ERROR_FUNCTION_NOT_FOUND);
			NVML_ERROR_CASE(NVML_ERROR_CORRUPTED_INFOROM);
			NVML_ERROR_CASE(NVML_ERROR_GPU_IS_LOST);
			NVML_ERROR_CASE(NVML_ERROR_RESET_REQUIRED);
			NVML_ERROR_CASE(NVML_ERROR_OPERATING_SYSTEM);
			NVML_ERROR_CASE(NVML_ERROR_LIB_RM_VERSION_MISMATCH);
			NVML_ERROR_CASE(NVML_ERROR_UNKNOWN);
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
} // cutf

#endif // __CUTF_NVML_HPP__
