#ifndef __CUTF_DEBUG_TF32_HPP__
#define __CUTF_DEBUG_TF32_HPP__
#include <cinttypes>

namespace cutf {
namespace experimental {
namespace tf32 {
namespace detail {
union to_bitstring {
	float fp;
	uint32_t bitstring;
};
union to_fp {
	uint32_t bitstring;
	float fp;
};
} // namespace detail

using tf32_t = float;

__device__ __host__ inline cutf::experimental::tf32::tf32_t to_tf32(const float v) {
	const uint32_t in = detail::to_bitstring{v}.bitstring;
	const uint32_t c0 = (in & 0b0'00000000'0000000000'1000000000000u);
	const uint32_t m = (in & 0b0'00000000'1111111111'0000000000000u);
	const uint32_t e = (in & 0b0'11111111'0000000000'0000000000000u);
	const uint32_t s = (in & 0b1'00000000'0000000000'0000000000000u);

	const uint32_t m0 = m + (c0 << 1);
	const uint32_t c1 = (m0 & 0b0'00000001'0000000000'0000000000000u) >> 23;
	const uint32_t m_pre = m0 & 0b0'00000000'1111111111'0000000000000u;
	const uint32_t e_pre = e + (c1 << 23);

	const uint32_t out = s | m_pre | e_pre;
	return detail::to_fp{out}.fp;
}

} // namespace tf32
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
