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
struct same_size_uint {using type = uint32_t;};
template <> struct same_size_uint<half  > {using type = uint16_t;};
template <> struct same_size_uint<float > {using type = uint32_t;};
template <> struct same_size_uint<double> {using type = uint64_t;};

template <class T>
struct same_size_fp {using type = float;};
template <> struct same_size_fp<uint16_t> {using type = half  ;};
template <> struct same_size_fp<uint32_t> {using type = float ;};
template <> struct same_size_fp<uint64_t> {using type = double;};

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
CUTF_DEVICE_HOST_FUNC inline unsigned get_bias();
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_bias<half  >() {return 0xf;}
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_bias<float >() {return 0x7f;}
template <> CUTF_DEVICE_HOST_FUNC inline unsigned get_bias<double>() {return 0x3ff;}

template <class T>
CUTF_DEVICE_HOST_FUNC inline typename same_size_uint<T>::type reinterpret_as_uint(const T fp) {
	return detail::reinterpret_medium<T, typename same_size_uint<T>::type>{.fp = fp}.bs;
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline typename same_size_fp<T>::type reinterpret_as_fp(const T bs) {
	return detail::reinterpret_medium<typename same_size_fp<T>::type, T>{.bs = bs}.fp;
}

template <>
CUTF_DEVICE_HOST_FUNC inline typename same_size_fp<uint16_t>::type reinterpret_as_fp<uint16_t>(const uint16_t bs) {
	return *reinterpret_cast<const half*>(&bs);
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline typename same_size_uint<T>::type mask_mantissa(const T fp) {
	const auto uint = cutf::experimental::fp::reinterpret_as_uint(fp);
	const auto mask = (decltype(uint)(1) << get_mantissa_size<T>()) - 1;
	return uint & mask;
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline typename same_size_uint<T>::type mask_exponent(const T fp) {
	const auto uint = cutf::experimental::fp::reinterpret_as_uint(fp);
	const auto mask = ((decltype(uint)(1) << get_exponent_size<T>()) - 1) << cutf::experimental::fp::get_mantissa_size<T>();
	return uint & mask;
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline typename same_size_uint<T>::type mask_sign(const T fp) {
	const auto uint = cutf::experimental::fp::reinterpret_as_uint(fp);
	const auto mask = decltype(uint)(1) << (sizeof(decltype(uint)) * 8 - 1);
	return uint & mask;
}
} // namespace fp
} // namespace experimental
} // namespace cutf
#endif
