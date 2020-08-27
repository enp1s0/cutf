#ifndef __CUTF_EXPERIMENTAL_FP_HPP__
#define __CUTF_EXPERIMENTAL_FP_HPP__
#include "../macro.hpp"
#include <cuda_fp16.h>

namespace cutf {
namespace experimental {
namespace fp {
template <class T>
CUTF_DEVICE_HOST_FUNC inline unsigned get_exponent_size();
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_exponent_size<half  >() {return 5;}
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_exponent_size<float >() {return 8;}
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_exponent_size<double>() {return 11;}

template <class T>
CUTF_DEVICE_HOST_FUNC inline unsigned get_mantissa_size();
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_mantissa_size<half  >() {return 10;}
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_mantissa_size<float >() {return 23;}
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_mantissa_size<double>() {return 52;}
} // namespace fp
} // namespace experimental
} // namespace cutf
#endif
