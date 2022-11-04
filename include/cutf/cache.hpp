#ifndef __CUTF_CACHE_HPP__
#define  __CUTF_CACHE_HPP__

namespace cutf {
namespace cache {
// cache residency and eviction mode
class L1;
class L2;
class L2_evict_last;
class L2_evict_normal;
// memory
class global;
class local;
// priority

template <class cache, class memory>
__device__ inline void prefetch(const void* const ptr);

template <>
__device__ inline void prefetch<L1, global>(const void* const ptr) {
	asm(R"({prefetch.global.L1 [%0];})" :: "l"(ptr));
}

template <>
__device__ inline void prefetch<L1, local>(const void* const ptr) {
	asm(R"({prefetch.local.L1 [%0];})" :: "l"(ptr));
}

template <>
__device__ inline void prefetch<L2, global>(const void* const ptr) {
	asm(R"({prefetch.global.L2 [%0];})" :: "l"(ptr));
}

template <>
__device__ inline void prefetch<L2, local>(const void* const ptr) {
	asm(R"({prefetch.local.L2 [%0];})" :: "l"(ptr));
}

template <>
__device__ inline void prefetch<L2_evict_normal, global>(const void* const ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm(R"({prefetch.global.L2::evict_normal [%0];})" :: "l"(ptr));
#endif
}

template <>
__device__ inline void prefetch<L2_evict_last, global>(const void* const ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
	asm(R"({prefetch.global.L2::evict_last [%0];})" :: "l"(ptr));
#endif
}

__device__ inline void prefetchu(const void* const ptr) {
	asm(R"({prefetchu.L1 [%0];})" :: "l"(ptr));
}
} // namespace cahce
} // namespace cutf
#endif

