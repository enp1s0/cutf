#ifndef __CUTF_CURAND_HPP__
#define __CUTF_CURAND_HPP__
#include <curand.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <string>
#include "error.hpp"

namespace cutf {
namespace error {
inline void check(curandStatus_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != CURAND_STATUS_SUCCESS){
		std::string error_string;
#define CURAND_ERROR_CASE(c) case c: error_string = #c; break
		switch(error){
			CURAND_ERROR_CASE(CURAND_STATUS_VERSION_MISMATCH);
			CURAND_ERROR_CASE(CURAND_STATUS_NOT_INITIALIZED);
			CURAND_ERROR_CASE(CURAND_STATUS_ALLOCATION_FAILED);
			CURAND_ERROR_CASE(CURAND_STATUS_TYPE_ERROR);
			CURAND_ERROR_CASE(CURAND_STATUS_OUT_OF_RANGE);
			CURAND_ERROR_CASE(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
			CURAND_ERROR_CASE(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
			CURAND_ERROR_CASE(CURAND_STATUS_LAUNCH_FAILURE);
			CURAND_ERROR_CASE(CURAND_STATUS_PREEXISTING_FAILURE);
			CURAND_ERROR_CASE(CURAND_STATUS_INITIALIZATION_FAILED);
			CURAND_ERROR_CASE(CURAND_STATUS_ARCH_MISMATCH);
			CURAND_ERROR_CASE(CURAND_STATUS_INTERNAL_ERROR);
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
namespace curand {
struct curand_deleter {
	void operator()(curandGenerator_t* cugen){
		cutf::error::check(curandDestroyGenerator(*cugen), __FILE__, __LINE__, __func__);
		delete cugen;
	}
};
inline std::unique_ptr<curandGenerator_t, curand_deleter> get_curand_unique_ptr(curandRngType_t rng_type){
	curandGenerator_t *cgen = new curandGenerator_t;
	curandCreateGenerator(cgen, rng_type);
	return std::unique_ptr<curandGenerator_t, curand_deleter>{cgen};
}
inline std::unique_ptr<curandGenerator_t, curand_deleter> get_curand_host_unique_ptr(curandRngType_t rng_type){
	curandGenerator_t *cgen = new curandGenerator_t;
	curandCreateGeneratorHost(cgen, rng_type);
	return std::unique_ptr<curandGenerator_t, curand_deleter>{cgen};
}

// normal distribution generator
inline curandStatus_t generate(curandGenerator_t gen, unsigned int* const output_ptr, const std::size_t size) {
	return curandGenerate(gen, output_ptr, size);
}
inline curandStatus_t generate(curandGenerator_t gen, unsigned long long* const output_ptr, const std::size_t size) {
	return curandGenerateLongLong(gen, output_ptr, size);
}

// normal distribution generator
inline curandStatus_t generate_normal(curandGenerator_t gen, float* const output_ptr, const std::size_t size, const float mean, const float stddev) {
	return curandGenerateNormal(gen, output_ptr, size, mean, stddev);
}
inline curandStatus_t generate_normal(curandGenerator_t gen, double* const output_ptr, const std::size_t size, const double mean, const double stddev) {
	return curandGenerateNormalDouble(gen, output_ptr, size, mean, stddev);
}

// log normal distribution generator
inline curandStatus_t generate_log_normal(curandGenerator_t gen, float* const output_ptr, const std::size_t size, const float mean, const float stddev) {
	return curandGenerateLogNormal(gen, output_ptr, size, mean, stddev);
}
inline curandStatus_t generate_log_normal(curandGenerator_t gen, double* const output_ptr, const std::size_t size, const double mean, const double stddev) {
	return curandGenerateLogNormalDouble(gen, output_ptr, size, mean, stddev);
}

// uniform distribution generator
inline curandStatus_t generate_uniform(curandGenerator_t gen, float* const output_ptr, const std::size_t size) {
	return curandGenerateUniform(gen, output_ptr, size);
}
inline curandStatus_t generate_uniform(curandGenerator_t gen, double* const output_ptr, const std::size_t size) {
	return curandGenerateUniformDouble(gen, output_ptr, size);
}
} // namespace curand
} // namespace cutf

#endif /* end of include guard */
