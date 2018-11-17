#ifndef __CUTF_ERROR_CUH__
#define __CUTF_ERROR_CUH__
#include <stdexcept>
#include <sstream>

namespace mtk{
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
} // mtk

#endif // __CUTF_ERROR_CUH__
