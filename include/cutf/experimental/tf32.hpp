#ifndef __CUTF_DEBUG_TF32_HPP__
#define __CUTF_DEBUG_TF32_HPP__
#include <cinttypes>
#include "../macro.hpp"
#include "mantissa.hpp"

namespace cutf {
namespace experimental {
namespace tf32 {

using tf32_t = float;

CUTF_DEVICE_HOST_FUNC inline cutf::experimental::tf32::tf32_t to_tf32(const float v) {
	return cutf::experimental::mantissa::cut_mantissa<10>(v);
}

} // namespace tf32
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
