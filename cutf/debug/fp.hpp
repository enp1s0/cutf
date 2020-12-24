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
__device__ __host__ inline void print_hex(const T v, const bool line_break = true);
template <> __device__ __host__ inline void print_hex<uint64_t>(const uint64_t v, const bool line_break) {printf("0x%016lx", v);if(line_break)printf("\n");}
template <> __device__ __host__ inline void print_hex<uint32_t>(const uint32_t v, const bool line_break) {printf("0x%08x", v);if(line_break)printf("\n");}
template <> __device__ __host__ inline void print_hex<uint16_t>(const uint16_t v, const bool line_break) {printf("0x%04x", v);if(line_break)printf("\n");}
template <> __device__ __host__ inline void print_hex<uint8_t >(const uint8_t  v, const bool line_break) {printf("0x%02x", v);if(line_break)printf("\n");}
template <> __device__ __host__ inline void print_hex<double  >(const double   v, const bool line_break) {print_hex(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
template <> __device__ __host__ inline void print_hex<float   >(const float    v, const bool line_break) {print_hex(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
template <> __device__ __host__ inline void print_hex<half    >(const half     v, const bool line_break) {print_hex(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}

template <class T>
__device__ __host__ inline void print_bin(const T v, const bool line_break = true) {
	for (int i = sizeof(T) * 8 - 1; i >= 0; i--) {
		printf("%d", static_cast<int>(v >> i) & 0x1);
	}
	if (line_break) {
		printf("\n");
	}
}
template <> __device__ __host__ inline void print_bin<half  >(const half   v, const bool line_break) {print_bin(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
template <> __device__ __host__ inline void print_bin<float >(const float  v, const bool line_break) {print_bin(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
template <> __device__ __host__ inline void print_bin<double>(const double v, const bool line_break) {print_bin(cutf::experimental::fp::reinterpret_as_uint(v), line_break);}
} // namespace print
} // namespace debug
} // namespace cutf
#endif
