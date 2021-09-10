#ifndef __CUTF_CUTENSOR_HPP__
#define __CUTF_CUTENSOR_HPP__
#include <string>
#include <sstream>
#include <cutensor.h>

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
} // cutf
#endif
