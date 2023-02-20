#ifndef __CUTF_DEBUG_FP_HPP__
#define __CUTF_DEBUG_FP_HPP__
#include <stdio.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "../experimental/fp.hpp"
namespace cutf {
namespace debug {
namespace fp {
template <class T>
struct bitstring_t {using type = T;};
template <> struct bitstring_t<half  > {using type = uint16_t;};
template <> struct bitstring_t<float > {using type = uint32_t;};
template <> struct bitstring_t<double> {using type = uint64_t;};
template <> struct bitstring_t<const double> {using type = uint64_t;};
template <> struct bitstring_t<const float > {using type = uint32_t;};
template <> struct bitstring_t<const half  > {using type = uint64_t;};
} // namespace fp

namespace print {
template <class T>
__device__ __host__ inline void print_hex(const T v, const bool line_break = true, const bool prefix = true);
template <> __device__ __host__ inline void print_hex<uint64_t>(const uint64_t v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%016lx\n", v);else printf("%016lx", v);}
template <> __device__ __host__ inline void print_hex<uint32_t>(const uint32_t v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%08x\n"  , v);else printf("%08x"  , v);}
template <> __device__ __host__ inline void print_hex<uint16_t>(const uint16_t v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%04x\n"  , v);else printf("%04x"  , v);}
template <> __device__ __host__ inline void print_hex<uint8_t >(const uint8_t  v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%02x\n"  , v);else printf("%02x"  , v);}
template <> __device__ __host__ inline void print_hex<int64_t >(const int64_t  v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%016lx\n", v);else printf("%016lx", v);}
template <> __device__ __host__ inline void print_hex<int32_t >(const int32_t  v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%08x\n"  , v);else printf("%08x"  , v);}
template <> __device__ __host__ inline void print_hex<int16_t >(const int16_t  v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%04x\n"  , v);else printf("%04x"  , v);}
template <> __device__ __host__ inline void print_hex<int8_t  >(const int8_t   v, const bool line_break, const bool prefix) {if(prefix)printf("0x");if(line_break)printf("%02x\n"  , v);else printf("%02x"  , v);}
template <> __device__ __host__ inline void print_hex<double  >(const double   v, const bool line_break, const bool prefix) {print_hex(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
template <> __device__ __host__ inline void print_hex<float   >(const float    v, const bool line_break, const bool prefix) {print_hex(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
template <> __device__ __host__ inline void print_hex<half    >(const half     v, const bool line_break, const bool prefix) {print_hex(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}

template <> __device__ __host__ inline void print_hex<__int128_t>(const __int128_t v, const bool line_break, const bool prefix) {
	union {
		std::int64_t v[2];
		__int128_t w;
	} a;
	a.w = v;
	print_hex(a.v[1], false, prefix);
	print_hex(a.v[0], line_break, false);
}

template <> __device__ __host__ inline void print_hex<__uint128_t>(const __uint128_t v, const bool line_break, const bool prefix) {
	union {
		std::uint64_t v[2];
		__uint128_t w;
	} a;
	a.w = v;
	print_hex(a.v[1], false, prefix);
	print_hex(a.v[0], line_break, false);
}

template <class T>
__device__ __host__ inline void print_bin(const T v, const bool line_break = true, const bool prefix = true) {
	if (prefix) {
		std::printf("0b");
	}
	char bs_str[cutf::experimental::fp::size_of<T>::value * 8 + 1] = {0};
	for (int i = sizeof(T) * 8 - 1; i >= 0; i--) {
		bs_str[cutf::experimental::fp::size_of<T>::value * 8 - 1 - i] = static_cast<char>(static_cast<int>('0') + (static_cast<int>(v >> i) & 0x1));
	}
	if (line_break) {
		printf("%s\n", bs_str);
	} else {
		printf("%s", bs_str);
	}
}
template <> __device__ __host__ inline void print_bin<half  >(const half   v, const bool line_break, const bool prefix) {print_bin(cutf::experimental::fp::reinterpret_as_uint(v), line_break, prefix);}
template <> __device__ __host__ inline void print_bin<float >(const float  v, const bool line_break, const bool prefix) {print_bin(cutf::experimental::fp::reinterpret_as_uint(v), line_break, prefix);}
template <> __device__ __host__ inline void print_bin<double>(const double v, const bool line_break, const bool prefix) {print_bin(cutf::experimental::fp::reinterpret_as_uint(v), line_break, prefix);}
} // namespace print
} // namespace debug
} // namespace cutf
#endif
