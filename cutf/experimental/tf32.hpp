#ifndef __CUTF_DEBUG_TF32_HPP__
#define __CUTF_DEBUG_TF32_HPP__
#include <cinttypes>
#include "mantissa.hpp"

#if !defined(CUTF_DEVICE_HOST) && defined(__CUDA_ARCH__)
#define CUTF_DEVICE_HOST CUTF_DEVICE_HOST
#else
#define CUTF_DEVICE_HOST
#endif

namespace cutf {
namespace experimental {
namespace tf32 {

using tf32_t = float;

CUTF_DEVICE_HOST inline cutf::experimental::tf32::tf32_t to_tf32(const float v) {
	return cutf::experimental::cut_mantissa<10>(v);
}

} // namespace tf32
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
