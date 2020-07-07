#ifndef __CUTF_DEBUG_TF32_HPP__
#define __CUTF_DEBUG_TF32_HPP__
#include <cinttypes>

namespace cutf {
namespace debug {
using tf32 = float;

__device__ __host__ inline cutf::debug::tf32 to_tf32(const float v) {
	const uint32_t in = *reinterpret_cast<const uint32_t*>(&v);
	const uint32_t c0 = (in & 0b0'00000000'0000000000'1000000000000u);
	const uint32_t m = (in & 0b0'00000000'1111111111'0000000000000u);
	const uint32_t e = (in & 0b0'11111111'0000000000'0000000000000u);
	const uint32_t s = (in & 0b1'00000000'0000000000'0000000000000u);

	const uint32_t m0 = m + (c0 << 1);
	const uint32_t c1 = (m0 & 0b0'00000001'0000000000'0000000000000u) >> 23;
	const uint32_t m_pre = m0 & 0b0'00000000'1111111111'0000000000000u;
	const uint32_t e_pre = e + (c1 << 23);

	const uint32_t out = s | m_pre | e_pre;
	return *reinterpret_cast<const float*>(&out);
}

} // namespace debug
} // namespace cutf

#endif /* end of include guard */
