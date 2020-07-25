#ifndef __CUTF_EXPERIMENTAL_MANTISSA_HPP__
#define __CUTF_EXPERIMENTAL_MANTISSA_HPP__
#include <cinttypes>

namespace cutf {
namespace experimental {
namespace detail {
union fp32_to_bitstring {
	float fp;
	uint32_t bitstring;
};
union bitstring_to_fp32 {
	uint32_t bitstring;
	float fp;
};
} // namespace detail
} // namespace experimental
} // namespace cutf
#endif
