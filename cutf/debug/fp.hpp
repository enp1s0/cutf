#ifndef __CUTF_DEBUG_FP_HPP__
#define __CUTF_DEBUG_FP_HPP__
#include <cuda_fp16.h>
#include <cstdint>
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
} // namespace debug
} // namespace cutf
#endif
