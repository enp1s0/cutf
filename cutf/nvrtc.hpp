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

namespace cutf{
namespace nvrtc{
namespace error{
inline void check(const nvrtcResult result, const std::string message,  const std::string filename, const std::size_t line, const std::string funcname){
	if(result != NVRTC_SUCCESS){
		std::stringstream ss;
		ss<< nvrtcGetErrorString( result ) << " : " << message <<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
} // error
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

	for(auto i = decltype(num_headers)(0); i < num_headers; i++){
		const auto header_name = headers[i].first;
		const auto header_source = headers[i].second;
		header_names.get()[i] = new char [header_name.length() + 1];
		header_sources.get()[i] = new char [header_source.length() + 1];
		std::strcpy(header_names.get()[i], header_name.c_str());
		std::strcpy(header_sources.get()[i], header_source.c_str());
	}

	nvrtcProgram program;
	error::check(nvrtcCreateProgram(
					&program,
					function_code.c_str(),
					source_name.c_str(),
					num_headers,
					header_sources.get(),
					header_names.get()
				), "@ Creating " + source_name, __FILE__, __LINE__, __func__);

	for(auto i = decltype(num_options)(0); i < num_options; i++){
		const auto option = compile_options[i];
		options.get()[i] = new char [option.length() + 1];
		std::strcpy(options.get()[i], option.c_str());
	}

	error::check(nvrtcCompileProgram(
				program,
				num_options,
				options.get()
				), "@ Compiling " + source_name, __FILE__, __LINE__, __func__);

	if( print_compile_log ){
		std::size_t log_size;
		error::check(nvrtcGetProgramLogSize(
					program,
					&log_size
					), "@ Getting compile log size of " + source_name, __FILE__, __LINE__, __func__);
		std::unique_ptr<char[]> log(new char[log_size]);
		error::check(nvrtcGetProgramLog(
					program,
					log.get()
					), "@ Getting compile log of " + source_name, __FILE__, __LINE__, __func__);
		std::cout<<log.get()<<std::endl;
	}

	std::size_t ptx_size;
	error::check(nvrtcGetPTXSize(
				program,
				&ptx_size
				), "@ Getting ptx size of " + source_name, __FILE__, __LINE__, __func__);
	std::unique_ptr<char[]> ptx_code(new char[ptx_size]);
	error::check(nvrtcGetPTX(
				program,
				ptx_code.get()), "@ Getting PTX code of " + source_name, __FILE__, __LINE__, __func__);
	error::check(nvrtcDestroyProgram(
				&program), " @ Destroying program of " + source_name, __FILE__, __LINE__, __func__);

	return std::string(ptx_code.get());
}
}
}

#endif // __CUTF_NVRTC__
