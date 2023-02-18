#ifndef __CUTF_CURAND_KERNEL_HPP__
#define __CUTF_CURAND_KERNEL_HPP__
#include "macro.hpp"
#include <curand_kernel.h>

namespace cutf {
namespace curand_kernel {
template <class T>
CUTF_DEVICE_FUNC inline T uniform(curandState_t* state);
template <> CUTF_DEVICE_FUNC inline float  uniform<float >(curandState_t* state) {return curand_uniform(state);};
template <> CUTF_DEVICE_FUNC inline double uniform<double>(curandState_t* state) {return curand_uniform_double(state);};

template <class T>
CUTF_DEVICE_FUNC inline T normal(curandState_t* state);
template <> CUTF_DEVICE_FUNC inline float   normal<float  >(curandState_t* state) {return curand_normal(state);};
template <> CUTF_DEVICE_FUNC inline double  normal<double >(curandState_t* state) {return curand_normal_double(state);};

template <class T>
CUTF_DEVICE_FUNC inline T log_normal(curandState_t* state, const T mean, const T stddev);
template <> CUTF_DEVICE_FUNC inline float   log_normal<float  >(curandState_t* state, float   mean, float   stddev) {return curand_log_normal        (state, mean, stddev);};
template <> CUTF_DEVICE_FUNC inline double  log_normal<double >(curandState_t* state, double  mean, double  stddev) {return curand_log_normal_double (state, mean, stddev);};
} // namespace curand_kernel
} // namespace cutf
#endif

