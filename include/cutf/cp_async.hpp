#ifndef __CUTF_CP_ASYNC_HPP__
#define __CUTF_CP_ASYNC_HPP__
#include <cstdint>
#include "macro.hpp"
namespace cutf {
namespace cp_async {
namespace detail {
CUTF_DEVICE_FUNC inline uint32_t get_smem_ptr_uint(const void* const ptr) {
  uint32_t smem_ptr;
  asm volatile("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n": "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
} // namespace detail

#if defined(CUTF_DISABLE_CP_ASYNC) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800)
#else
#define CUTF_INTERNAL_USE_CP_ASYNC
#endif

template <unsigned Size>
CUTF_DEVICE_FUNC inline void cp_async(void* const smem, const void* const gmem) {
	static_assert(Size == 4 || Size == 8 || Size == 16, "Size must be one of 4, 8 and 16");
#ifdef CUTF_INTERNAL_USE_CP_ASYNC
	const unsigned smem_int_ptr = detail::get_smem_ptr_uint(smem);
	asm volatile("{cp.async.ca.shared.global [%0], [%1], %2;}" :: "r"(smem_int_ptr), "l"(gmem), "n"(Size));
#else
	if (Size == 4) {
		*(reinterpret_cast<uint32_t*>(smem)) = *(reinterpret_cast<const uint32_t*>(gmem));
	} else if (Size == 8) {
		*(reinterpret_cast<uint64_t*>(smem)) = *(reinterpret_cast<const uint64_t*>(gmem));
	} else {
		*(reinterpret_cast<ulong2*>(smem)) = *(reinterpret_cast<const ulong2*>(gmem));
	}
#endif
}

CUTF_DEVICE_FUNC inline void commit() {
#ifdef CUTF_INTERNAL_USE_CP_ASYNC
	asm volatile("{cp.async.commit_group;}\n");
#endif
}

CUTF_DEVICE_FUNC inline void wait_all() {
#ifdef CUTF_INTERNAL_USE_CP_ASYNC
	asm volatile("{cp.async.wait_all;}");
#endif
}

template <int N>
CUTF_DEVICE_FUNC inline void wait_group() {
#ifdef CUTF_INTERNAL_USE_CP_ASYNC
	asm volatile("{cp.async.wait_group %0;}":: "n"(N));
#endif
}

} // namespace cp_async
} // namespace cutf
#endif
