#ifndef __CUTF_DEBUG_FRAGMENT_HPP__
#define __CUTF_DEBUG_FRAGMENT_HPP__
#include <stdio.h>
#include <mma.h>
#include "../thread.hpp"
#include "../type.hpp"

namespace cutf {
namespace debug {
namespace print {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 7000
template <class Use, int M, int N, int K, class T, class Layout>
__device__ void print_fragment(const nvcuda::wmma::fragment<Use, M, N, K, T, Layout>& fragment, const char* const name = nullptr) {
	if (name != nullptr && cutf::thread::get_lane_id() == 0) {
		printf("%s =\n", name);
	}
	for (unsigned tid = 0; tid < warpSize; tid++) {
		if (tid == cutf::thread::get_lane_id()) {
			for (unsigned i = 0; i < fragment.num_elements; i++) {
				printf("%+e ", cutf::type::cast<float>(fragment.x[i]));
			}
			printf("\n");
		}
		__syncthreads();
	}
}
#else
#endif
} // namespace print
} // namespace debug
} // namespace cutf
#endif
