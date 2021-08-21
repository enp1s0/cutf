#ifndef __CUTF_NVRTC__
#define __CUTF_NVRTC__
#include <string>
#include <cstring>
#include <vector>
#include <utility>
#include <memory>
#include <iostream>
#include <nvrtc.h>
#include <cuda.h>
#include "cuda.hpp"
#include "driver.hpp"
#include "error.hpp"

namespace cutf{
namespace error{
inline void check(const nvrtcResult result,  const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(result != NVRTC_SUCCESS){
		std::stringstream ss;
		ss<< nvrtcGetErrorString( result );
		if(message.length() != 0){
			ss<<" : "<<message;
		}
	    ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
} // error
namespace nvrtc{
inline std::string get_ptx(
		const std::string source_name, 
		const std::string function_code, 
		const std::vector<std::string> compile_options = {}, 
		const std::vector<std::pair<std::string, std::string>> headers = {},
		const bool print_compile_log = false
		){
	const std::size_t num_headers = headers.size();
	const std::size_t num_options = headers.size();
	std::unique_ptr<char* []> options(new char*[num_options]);
	std::unique_ptr<char* []> header_names(new char*[num_headers]);
	std::unique_ptr<char* []> header_sources(new char*[num_headers]);

	// Creating Program 1{{{
	for(auto i = decltype(num_headers)(0); i < num_headers; i++){
		const auto header_name = headers[i].first;
		const auto header_source = headers[i].second;
		header_names.get()[i] = new char [header_name.length() + 1];
		header_sources.get()[i] = new char [header_source.length() + 1];
		std::strcpy(header_names.get()[i], header_name.c_str());
		std::strcpy(header_sources.get()[i], header_source.c_str());
	}

	nvrtcProgram program;
	cutf::error::check(nvrtcCreateProgram(
					&program,
					function_code.c_str(),
					source_name.c_str(),
					num_headers,
					header_sources.get(),
					header_names.get()
				), __FILE__, __LINE__, __func__, "@ Creating " + source_name);

	for(auto i = decltype(num_headers)(0); i < num_headers; i++){
		delete [] header_names.get()[i];
		delete [] header_sources.get()[i];
	}
	/// }}}1

	// Compiling 2{{{
	for(auto i = decltype(num_options)(0); i < num_options; i++){
		const auto option = compile_options[i];
		options.get()[i] = new char [option.length() + 1];
		std::strcpy(options.get()[i], option.c_str());
	}

	cutf::error::check(nvrtcCompileProgram(
				program,
				num_options,
				options.get()
				), __FILE__, __LINE__, __func__, "@ Compiling " + source_name);

	for(auto i = decltype(num_options)(0); i < num_options; i++){
		delete [] options.get()[i];
	}
	// }}}2

	// Printing log 3{{{
	if( print_compile_log ){
		std::size_t log_size;
		error::check(nvrtcGetProgramLogSize(
					program,
					&log_size
					), __FILE__, __LINE__, __func__, "@ Getting compile log size of " + source_name);
		std::unique_ptr<char[]> log(new char[log_size]);
		error::check(nvrtcGetProgramLog(
					program,
					log.get()
					), __FILE__, __LINE__, __func__, "@ Getting compile log of " + source_name);
		std::cout<<log.get()<<std::endl;
	}
	/// }}}3

	// Getting PTX 4{{{
	std::size_t ptx_size;
	cutf::error::check(nvrtcGetPTXSize(
				program,
				&ptx_size
				), __FILE__, __LINE__, __func__, "@ Getting ptx size of " + source_name);
	std::unique_ptr<char[]> ptx_code(new char[ptx_size]);
	cutf::error::check(nvrtcGetPTX(
				program,
				ptx_code.get()), __FILE__, __LINE__, __func__, "@ Getting PTX code of " + source_name);

	// }}}4

	cutf::error::check(nvrtcDestroyProgram(
				&program), __FILE__, __LINE__, __func__, " @ Destroying program of " + source_name);

	return std::string(ptx_code.get());
}

inline CUfunction get_function(
		const std::string ptx_code,
		const std::string function_name,
		CUmodule *cu_module,
		const unsigned int device_id = 0
		){
	CUfunction function;
	cutf::error::check(cuModuleLoadDataEx(cu_module, ptx_code.c_str(), 0, 0, 0), __FILE__, __LINE__, __func__, "@ Loading module(ptx) " + function_name);
	cutf::error::check(cuModuleGetFunction(&function, *cu_module, function_name.c_str()), __FILE__, __LINE__, __func__, "@ Getting function " + function_name);

	return function;
}

inline void launch_function(
		const CUfunction function,
		std::vector<void*> arguments_pointers,
		const dim3 grid,
		const dim3 block,
		CUstream stream = nullptr,
		unsigned int shared_memory_size = 0
		){
	cutf::error::check(cuLaunchKernel(
					function,
					grid.x, grid.y, grid.z,
					block.x, block.y, block.z,
					shared_memory_size,
					stream,
					arguments_pointers.data(),
					nullptr
				), __FILE__, __LINE__, __func__);
}
} // nvrtc
} // cutf

#endif // __CUTF_NVRTC__
