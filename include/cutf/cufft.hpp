#ifndef __CUTF_CUFTT_HPP__
#define __CUTF_CUFTT_HPP__
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <cufftXt.h>
#include <memory>
#include <complex>
#include "error.hpp"

namespace cutf {
namespace error {
inline void check(const cufftResult error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CUFFT_SUCCESS){
		std::string error_string;
#define CUFFT_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
		CUFFT_ERROR_CASE(CUFFT_SUCCESS                  );
    CUFFT_ERROR_CASE(CUFFT_INVALID_PLAN             );
    CUFFT_ERROR_CASE(CUFFT_ALLOC_FAILED             );
    CUFFT_ERROR_CASE(CUFFT_INVALID_TYPE             );
    CUFFT_ERROR_CASE(CUFFT_INVALID_VALUE            );
    CUFFT_ERROR_CASE(CUFFT_INTERNAL_ERROR           );
    CUFFT_ERROR_CASE(CUFFT_EXEC_FAILED              );
    CUFFT_ERROR_CASE(CUFFT_SETUP_FAILED             );
    CUFFT_ERROR_CASE(CUFFT_INVALID_SIZE             );
    CUFFT_ERROR_CASE(CUFFT_UNALIGNED_DATA           );
    CUFFT_ERROR_CASE(CUFFT_INCOMPLETE_PARAMETER_LIST);
    CUFFT_ERROR_CASE(CUFFT_INVALID_DEVICE           );
    CUFFT_ERROR_CASE(CUFFT_PARSE_ERROR              );
    CUFFT_ERROR_CASE(CUFFT_NO_WORKSPACE             );
    CUFFT_ERROR_CASE(CUFFT_NOT_IMPLEMENTED          );
    CUFFT_ERROR_CASE(CUFFT_LICENSE_ERROR            );
    CUFFT_ERROR_CASE(CUFFT_NOT_SUPPORTED            );
		default: error_string = "Unknown error"; break;
		}
#undef CUFFT_ERROR_CASE
		std::stringstream ss;
		ss << error_string;
		if(message.length() != 0){
			ss<<" : "<<message;
		}
		ss<<" ["<<filename<<":"<<line<<" in "<<funcname<<"]";
		throw std::runtime_error(ss.str());
	}
}
}
namespace cufft {
struct cufft_deleter{
	void operator()(cufftHandle* handle){
		cutf::error::check(cufftDestroy(*handle), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
inline std::unique_ptr<cufftHandle, cufft_deleter> get_handle_unique_ptr(){
	auto *handle = new cufftHandle;
	CUTF_CHECK_ERROR(cufftCreate(handle));
	return std::unique_ptr<cufftHandle, cufft_deleter>{handle};
}

template <class INPUT_T, class OUTPUT_T>
inline cufftType get_type();
#define CUTF_CUFFT_TYPE(src_t, dst_t, mode) \
template <> inline cufftType get_type<src_t, dst_t>() {return mode;};
CUTF_CUFFT_TYPE(float          , cuComplex      , CUFFT_R2C);
CUTF_CUFFT_TYPE(cuComplex      , cuComplex      , CUFFT_C2C);
CUTF_CUFFT_TYPE(cuComplex      , float          , CUFFT_C2R);
CUTF_CUFFT_TYPE(double         , cuDoubleComplex, CUFFT_D2Z);
CUTF_CUFFT_TYPE(cuDoubleComplex, cuDoubleComplex, CUFFT_Z2Z);
CUTF_CUFFT_TYPE(cuDoubleComplex, double         , CUFFT_Z2D);
CUTF_CUFFT_TYPE(float               , std::complex<float> , CUFFT_R2C);
CUTF_CUFFT_TYPE(std::complex<float> , std::complex<float> , CUFFT_C2C);
CUTF_CUFFT_TYPE(std::complex<float> , float               , CUFFT_C2R);
CUTF_CUFFT_TYPE(double              , std::complex<double>, CUFFT_D2Z);
CUTF_CUFFT_TYPE(std::complex<double>, std::complex<double>, CUFFT_Z2Z);
CUTF_CUFFT_TYPE(std::complex<double>, double              , CUFFT_Z2D);
#undef CUTF_CUFFT_TYPE
} // namespace cufft
} // namespace cutf
#endif
