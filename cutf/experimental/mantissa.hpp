#ifndef __CUTF_EXPERIMENTAL_MANTISSA_HPP__
#define __CUTF_EXPERIMENTAL_MANTISSA_HPP__
#include <cinttypes>
#include "../rounding_mode.hpp"
#include "../macro.hpp"
#include "fp.hpp"

namespace cutf {
namespace experimental {
namespace mantissa {
namespace detail {

template <class T>
CUTF_DEVICE_HOST_FUNC inline T adjust_mantissa(const T mantissa, const T mantissa_mask, const uint32_t carry_bit, T& move_up) {
	move_up = (mantissa >> carry_bit) & 0x1;
	return mantissa & mantissa_mask;
}

template <class rounding>
CUTF_DEVICE_HOST_FUNC inline uint32_t rounding_mantissa(const uint32_t fp_bitstring, const uint32_t cut_length, uint32_t &move_up);

template <>
CUTF_DEVICE_HOST_FUNC inline uint32_t rounding_mantissa<cutf::rounding::rz>(const uint32_t fp_bitstring, const uint32_t cut_length, uint32_t &move_up) {
	move_up = 0;
	return (fp_bitstring & (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
}

template <>
CUTF_DEVICE_HOST_FUNC inline uint32_t rounding_mantissa<cutf::rounding::rr>(const uint32_t fp_bitstring, const uint32_t cut_length, uint32_t &move_up) {
	const uint32_t m0 = (fp_bitstring & (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
	const uint32_t c0 = (fp_bitstring & (1u << (cut_length - 1)));
	const uint32_t m1 = m0 + (c0 << 1);
	const uint32_t m_pre = adjust_mantissa(m1, (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)), 23, move_up);
	return m_pre;
}

template <>
CUTF_DEVICE_HOST_FUNC inline uint32_t rounding_mantissa<cutf::rounding::rn>(const uint32_t fp_bitstring, const uint32_t cut_length, uint32_t &move_up) {
	const uint32_t m0 = (fp_bitstring & (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
	const uint32_t c0 = (fp_bitstring & (1u << cut_length));
	const uint32_t m1 = m0 + c0;
	const uint32_t m_pre = adjust_mantissa(m1, (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)), 23, move_up);
	return m_pre;
}

template <>
CUTF_DEVICE_HOST_FUNC inline uint32_t rounding_mantissa<cutf::rounding::rb>(const uint32_t fp_bitstring, const uint32_t cut_length, uint32_t &move_up) {
	const uint32_t m0 = (fp_bitstring & (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)));
	const uint32_t m0_res = (fp_bitstring & ((1u << cut_length) - 1));
	uint32_t c0 = 0;
	if (m0_res != 0) {
		c0 = 1u << cut_length;
	}
	const uint32_t m1 = m0 + c0;
	const uint32_t m_pre = adjust_mantissa(m1, (0b0'00000000'1111111111'1111111111111u - ((1u << cut_length) - 1)), 23, move_up);
	return m_pre;
}

template <class rounding>
CUTF_DEVICE_HOST_FUNC inline uint64_t rounding_mantissa(const uint64_t fp_bitstring, const uint64_t cut_length, uint64_t &move_up);

template <>
CUTF_DEVICE_HOST_FUNC inline uint64_t rounding_mantissa<cutf::rounding::rz>(const uint64_t fp_bitstring, const uint64_t cut_length, uint64_t &move_up) {
	move_up = 0;
	return (fp_bitstring & (0x000ffffffffffffflu - ((1llu << cut_length) - 1)));
}

template <>
CUTF_DEVICE_HOST_FUNC inline uint64_t rounding_mantissa<cutf::rounding::rr>(const uint64_t fp_bitstring, const uint64_t cut_length, uint64_t &move_up) {
	const uint64_t m0 = (fp_bitstring & (0x000ffffffffffffflu - ((1llu << cut_length) - 1)));
	const uint64_t c0 = (fp_bitstring & (1u << (cut_length - 1)));
	const uint64_t m1 = m0 + (c0 << 1);
	const uint64_t m_pre = adjust_mantissa(m1, (0x000ffffffffffffflu - ((1lu << cut_length) - 1)), 53, move_up);
	return m_pre;
}

template <>
CUTF_DEVICE_HOST_FUNC inline uint64_t rounding_mantissa<cutf::rounding::rn>(const uint64_t fp_bitstring, const uint64_t cut_length, uint64_t &move_up) {
	const uint64_t m0 = (fp_bitstring & (0x000ffffffffffffflu - ((1llu << cut_length) - 1)));
	const uint64_t c0 = (fp_bitstring & (1u << cut_length));
	const uint64_t m1 = m0 + c0;
	const uint64_t m_pre = adjust_mantissa(m1, (0x000ffffffffffffflu - ((1lu << cut_length) - 1)), 53, move_up);
	return m_pre;
}

template <>
CUTF_DEVICE_HOST_FUNC inline uint64_t rounding_mantissa<cutf::rounding::rb>(const uint64_t fp_bitstring, const uint64_t cut_length, uint64_t &move_up) {
	const uint64_t m0 = (fp_bitstring & (0x000ffffffffffffflu - ((1llu << cut_length) - 1)));
	const uint64_t m0_res = (fp_bitstring & ((1lu << cut_length) - 1));
	uint64_t c0 = 0;
	if (m0_res != 0) {
		c0 = 1lu << cut_length;
	}
	const uint64_t m1 = m0 + c0;
	const uint64_t m_pre = adjust_mantissa(m1, (0x000ffffffffffffflu - ((1lu << cut_length) - 1)), 53, move_up);
	return m_pre;
}
} // namespace detail

template <unsigned mantissa_length, class rounding = cutf::rounding::rr>
CUTF_DEVICE_HOST_FUNC inline float cut_mantissa(const float v) {
	static_assert(mantissa_length > 0, "mantissa_length must be greater than 0");
	static_assert(mantissa_length < 23, "mantissa_length must be smaller than 23");

	constexpr unsigned cut_length = 23u - mantissa_length;
	const uint32_t in = cutf::experimental::fp::reinterpret_as_uint(v);
	const uint32_t e = (in & 0b0'11111111'00000000000000000000000u);
	const uint32_t s = (in & 0b1'00000000'00000000000000000000000u);

	uint32_t c1;
	const uint32_t m_pre = detail::rounding_mantissa<rounding>(in, cut_length, c1);
	const uint32_t e_pre = e + (c1 << 23);

	const uint32_t out = s | m_pre | e_pre;
	return cutf::experimental::fp::reinterpret_as_fp(out);
}

template <unsigned mantissa_length, class rounding = cutf::rounding::rr>
CUTF_DEVICE_HOST_FUNC inline double cut_mantissa(const double v) {
	static_assert(mantissa_length > 0, "mantissa_length must be greater than 0");
	static_assert(mantissa_length < 52, "mantissa_length must be smaller than 52");

	constexpr unsigned cut_length = 52u - mantissa_length;
	const uint64_t in = cutf::experimental::fp::reinterpret_as_uint(v);
	const uint64_t e = (in & 0xfff0000000000000lu);
	const uint64_t s = (in & 0x8000000000000000lu);

	uint64_t c1;
	const uint64_t m_pre = detail::rounding_mantissa<rounding>(in, cut_length, c1);
	const uint64_t e_pre = e + (c1 << 52);

	const uint64_t out = s | m_pre | e_pre;
	return cutf::experimental::fp::reinterpret_as_fp(out);
}
} // namespace mantissa
} // namespace experimental
} // namespace cutf
#endif
