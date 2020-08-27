#ifndef __CUTF_EXPERIMENTAL_EXPONENT_HPP__
#define __CUTF_EXPERIMENTAL_EXPONENT_HPP__
#include <cstdint>
#include "fp.hpp"
#include "../macro.hpp"

namespace cutf {
namespace experimental {
namespace exponent {
template <class T, int min_exponent>
CUTF_DEVICE_HOST_FUNC T force_underflow(const T v) {
	const auto bitstring = cutf::experimental::fp::reinterpret_as_uint(v);
	const auto exponent = ((bitstring << 1) >> (1 + cutf::experimental::fp::get_mantissa_size<T>()));
	const auto sp_exponent = static_cast<int>(exponent) - static_cast<int>(cutf::experimental::fp::get_bias<T>());
	if (sp_exponent < min_exponent) {
		// multiply zero to keep sign
		return v * 0;
	}
	return v;
}
} // namespace exponent
} // namespace experimental
} // namespace cutf
#endif
