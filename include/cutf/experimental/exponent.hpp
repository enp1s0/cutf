#ifndef __CUTF_EXPERIMENTAL_EXPONENT_HPP__
#define __CUTF_EXPERIMENTAL_EXPONENT_HPP__
#include <cstdint>
#include "fp.hpp"
#include "../macro.hpp"
#include "../type.hpp"

namespace cutf {
namespace experimental {
namespace exponent {
template <class T>
CUTF_DEVICE_HOST_FUNC T min_exponent(const T v, const int min_e) {
	const auto bitstring = cutf::experimental::fp::reinterpret_as_uint(v);
	const auto exponent = ((bitstring << 1) >> (1 + cutf::experimental::fp::get_mantissa_size<T>()));
	const auto sp_exponent = static_cast<int>(exponent) - static_cast<int>(cutf::experimental::fp::get_bias<T>());
	if (sp_exponent < min_e) {
		// multiply zero to keep sign
		return v * cutf::type::cast<T>(0);
	}
	return v;
}
} // namespace exponent
} // namespace experimental
} // namespace cutf
#endif
