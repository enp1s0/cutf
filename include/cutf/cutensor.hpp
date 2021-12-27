#ifndef __CUTF_CUTENSOR_HPP__
#define __CUTF_CUTENSOR_HPP__
#include <string>
#include <sstream>
#include <cutensor.h>
#include "type.hpp"

namespace cutf {
namespace error {
inline void check(cutensorStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUTENSOR_STATUS_SUCCESS){
		std::string error_string = cutensorGetErrorString(error);
		std::stringstream ss;
		ss << error_string;
		if(message.length() != 0){
			ss<<" : "<<message;
		}
		ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
} // error
namespace cutensor {
template <class T>
cutensorComputeType_t get_compute_type();
template <> cutensorComputeType_t get_compute_type<double                       >() {return CUTENSOR_COMPUTE_64F;}
template <> cutensorComputeType_t get_compute_type<float                        >() {return CUTENSOR_COMPUTE_32F;}
template <> cutensorComputeType_t get_compute_type<half                         >() {return CUTENSOR_COMPUTE_16F;}
template <> cutensorComputeType_t get_compute_type<nvcuda::wmma::precision::tf32>() {return CUTENSOR_COMPUTE_TF32;}
template <> cutensorComputeType_t get_compute_type<__nv_bfloat16                >() {return CUTENSOR_COMPUTE_16BF;}
template <> cutensorComputeType_t get_compute_type<uint32_t                     >() {return CUTENSOR_COMPUTE_32U;}
template <> cutensorComputeType_t get_compute_type<int32_t                      >() {return CUTENSOR_COMPUTE_32I;}
template <> cutensorComputeType_t get_compute_type<uint8_t                      >() {return CUTENSOR_COMPUTE_8U;}
template <> cutensorComputeType_t get_compute_type<int8_t                       >() {return CUTENSOR_COMPUTE_8I;}
} // namespace cutensor
} // namespace cutf
#endif
