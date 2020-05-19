#ifndef __CUTF_ERROR_CUH__
#define __CUTF_ERROR_CUH__
#include <stdexcept>
#include <sstream>
#include <memory>
#include <cuda_device_runtime_api.h>
#include <cuda.h>

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

namespace cu {
struct cumodule_deleter{
	void operator()(CUmodule* cumodule){
		cutf::error::check(cuModuleUnload(*cumodule), __FILE__, __LINE__, __func__);
		delete cumodule;
	}
};
inline std::unique_ptr<CUmodule, cumodule_deleter> get_module_unique_ptr(){
	CUmodule *cumodule= new CUmodule;
	return std::unique_ptr<CUmodule, cumodule_deleter>{cumodule};
}

struct cucontext_deleter{
	void operator()(CUcontext* cucontext){
		cutf::error::check(cuCtxDestroy(*cucontext), __FILE__, __LINE__, __func__);
		delete cucontext;
	}
};
inline std::unique_ptr<CUcontext, cucontext_deleter> get_context_unique_ptr(){
	CUcontext *cucontext= new CUcontext;
	return std::unique_ptr<CUcontext, cucontext_deleter>{cucontext};
}
} // namespace cu
} // cutf

#endif // __CUTF_ERROR_CUH__
