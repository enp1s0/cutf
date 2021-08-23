#ifndef __CUTF_CP_ASYNC_HPP__
#define __CUTF_CP_ASYNC_HPP__
namespace cutf {
namespace cp_async {
namespace detail {
__device__ inline uint32_t get_smem_ptr_uint(const void* const ptr) {
  uint32_t smem_ptr;
  asm volatile("{.reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }\n": "=r"(smem_ptr) : "l"(ptr));
  return smem_ptr;
}
} // namespace detail

template <unsigned Size>
__device__ inline void cp_async(void* const smem, const void* const gmem) {
#if __CUDA_ARCH__ >= 800
	static_assert(Size == 4 || Size == 8 || Size == 16, "Size must be one of 4, 8 and 16");
	const unsigned smem_int_ptr = detail::get_smem_ptr_uint(smem);
	asm volatile("{cp.async.ca.shared.global [%0], [%1], %2;}" :: "r"(smem_int_ptr), "l"(gmem), "n"(Size));
#else
	for (unsigned i = 0; i < Size / 4; i++) {
		*(reinterpret_cast<uint32_t*>(smem) + i) = *(reinterpret_cast<const uint32_t*>(gmem) + i);
	}
#endif
}

__device__ inline void commit() {
#if __CUDA_ARCH__ >= 800
	asm volatile("{cp.async.commit_group;}\n");
#endif
}

__device__ inline void wait_all() {
#if __CUDA_ARCH__ >= 800
	asm volatile("{cp.async.wait_all;}");
#endif
}

template <int N>
__device__ inline void wait_group() {
#if __CUDA_ARCH__ >= 800
	asm volatile("{cp.async.wait_group %0;}":: "n"(N));
#endif
}

} // namespace cp_async
} // namespace cutf
#endif
