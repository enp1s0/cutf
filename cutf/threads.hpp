#ifndef __CUTF_THREADS_HPP__
#define __CUTF_THREADS_HPP__

namespace cutf {
namespace threads {
__device__ inline unsigned get_lane_id() {
	unsigned warp_id;
	asm(R"({mov.s32 %0, %warpid;})":"=r"(warp_id));
	return warp_id;
}
} // namespace threads
} // namespace cutf
#endif
