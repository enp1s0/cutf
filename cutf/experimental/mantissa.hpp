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
union fp64_to_bitstring {
	double fp;
	uint64_t bitstring;
};
union bitstring_to_fp64 {
	uint64_t bitstring;
	double fp;
};
} // namespace detail

template <unsigned mantissa_length>
__device__ __host__ inline float cut_mantissa(const float v) {
	static_assert(mantissa_length > 0, "mantissa_length must be greater than 0");
	static_assert(mantissa_length < 23, "mantissa_length must be smaller than 23");

	constexpr unsigned cut_length = 23u - mantissa_length;
	const uint32_t in = cutf::experimental::detail::fp32_to_bitstring{v}.bitstring;
	const uint32_t c0 = (in & (1u << (cut_length - 1)));
	const uint32_t m = (in & (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
	const uint32_t e = (in & 0b0'11111111'00000000000000000000000u);
	const uint32_t s = (in & 0b1'00000000'00000000000000000000000u);

	const uint32_t m0 = m + (c0 << 1);
	const uint32_t c1 = (m0 & 0b0'00000001'00000000000000000000000u) >> 23;
	const uint32_t m_pre = m0 & (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1));
	const uint32_t e_pre = e + (c1 << 23);

	const uint32_t out = s | m_pre | e_pre;
	return cutf::experimental::detail::bitstring_to_fp32{out}.fp;
}

template <unsigned mantissa_length>
__device__ __host__ inline double cut_mantissa(const double v) {
	static_assert(mantissa_length > 0, "mantissa_length must be greater than 0");
	static_assert(mantissa_length < 52, "mantissa_length must be smaller than 52");

	constexpr unsigned cut_length = 52u - mantissa_length;
	const uint64_t in = cutf::experimental::detail::fp64_to_bitstring{v}.bitstring;
	const uint64_t c0 = (in & (1u << (cut_length - 1)));
	const uint64_t m = (in & (0x000ffffffffffffflu - ((1llu << cut_length) - 1)));
	const uint64_t e = (in & 0xfff0000000000000lu);
	const uint64_t s = (in & 0x8000000000000000lu);

	const uint64_t m0 = m + (c0 << 1);
	const uint64_t c1 = (m0 & 0x0010000000000000lu) >> 52;
	const uint64_t m_pre = m0 & (0x000ffffffffffffflu - ((1lu << cut_length) - 1));
	const uint64_t e_pre = e + (c1 << 52);

	const uint64_t out = s | m_pre | e_pre;
	return cutf::experimental::detail::bitstring_to_fp64{out}.fp;
}
} // namespace experimental
} // namespace cutf
#endif
