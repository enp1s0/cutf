#ifndef __CUTF_EXPERIMENTAL_FP_HPP__
#define __CUTF_EXPERIMENTAL_FP_HPP__
#include "../macro.hpp"
#include <cuda_fp16.h>
#include <cstdint>

namespace cutf {
namespace experimental {
namespace fp {
namespace detail {
template <class FP_T, class BS_T>
union reinterpret_medium {
	FP_T fp;
	BS_T bs;
};
} // namespace detail

template <class T>
struct same_size_uint {using type = void;};
template <> struct same_size_uint<half  > {using type = uint16_t;};
template <> struct same_size_uint<float > {using type = uint32_t;};
template <> struct same_size_uint<double> {using type = uint64_t;};

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

template <class T>
CUTF_DEVICE_HOST_FUNC inline typename same_size_uint<T>::type reinterpret_as_uint(const T fp) {
	return reinterpret_medium<T, typename same_size_uint<T>::type>{.fp = fp;}.bs;
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline T reinterpret_as_fp(const typename same_size_uint<T>::type bs) {
	return reinterpret_medium<T, typename same_size_uint<T>::type>{.bs = bs;}.fp;
}
} // namespace fp
} // namespace experimental
} // namespace cutf
#endif
