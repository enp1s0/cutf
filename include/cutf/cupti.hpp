#ifndef __CUTF_CUPTI_HPP__
#define __CUTF_CUPTI_HPP__
#include <cupti.h>

namespace cutf {
namespace error {
inline void check(const CUptiResult error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUPTI_SUCCESS){
		std::string error_string;
#define CUPTI_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_PARAMETER);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_DEVICE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_CONTEXT);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_EVENT_DOMAIN_ID);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_EVENT_ID);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_EVENT_NAME);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_OPERATION);
			CUPTI_ERROR_CASE(CUPTI_ERROR_OUT_OF_MEMORY);
			CUPTI_ERROR_CASE(CUPTI_ERROR_HARDWARE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_PARAMETER_SIZE_NOT_SUFFICIENT);
			CUPTI_ERROR_CASE(CUPTI_ERROR_API_NOT_IMPLEMENTED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_MAX_LIMIT_REACHED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_NOT_READY);
			CUPTI_ERROR_CASE(CUPTI_ERROR_NOT_COMPATIBLE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_NOT_INITIALIZED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_METRIC_ID);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_METRIC_NAME);
			CUPTI_ERROR_CASE(CUPTI_ERROR_QUEUE_EMPTY);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_HANDLE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_STREAM);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_KIND);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_EVENT_VALUE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_DISABLED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_MODULE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_INVALID_METRIC_VALUE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_HARDWARE_BUSY);
			CUPTI_ERROR_CASE(CUPTI_ERROR_NOT_SUPPORTED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE);
			CUPTI_ERROR_CASE(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES);
			CUPTI_ERROR_CASE(CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_WITH_MPS);
			CUPTI_ERROR_CASE(CUPTI_ERROR_CDP_TRACING_NOT_SUPPORTED);
			CUPTI_ERROR_CASE(CUPTI_ERROR_UNKNOWN);
			CUPTI_ERROR_CASE(CUPTI_ERROR_FORCE_INT);
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
} // namespace error
} // namespace cutf
#endif
