#ifndef __CUTF_MEMORY_CUH__
#define __CUTF_MEMORY_CUH__

#include <memory>
#include "error.cuh"

#define MTK_CUDA_CHECK_ERROR(error_code) mtk::cuda::error::check( error_code, __FILE__, __LINE__, __func__)

namespace mtk{
namespace cuda{
namespace memory{

// deleter
template <class T>
class device_deleter{
public:
	void operator()(T* ptr){
		MTK_CUDA_CHECK_ERROR(cudaFree( ptr ));
	}
};
template <class T>
class host_deleter{
public:
	void operator()(T* ptr){
		MTK_CUDA_CHECK_ERROR(cudaFreeHost( ptr ));
	}
};

// unique pointer type for c++ 11
template <class T>
using device_unique_ptr = std::unique_ptr<T, device_deleter<T>>;
template <class T>
using host_unique_ptr = std::unique_ptr<T, host_deleter<T>>;

// allocater
template <class T>
inline device_unique_ptr<T> get_device_unique_ptr(const std::size_t size){
	T* ptr;
	MTK_CUDA_CHECK_ERROR(cudaMalloc((void**)&ptr, sizeof(T) * size));
	return std::unique_ptr<T, device_deleter<T>>{ptr};
}
template <class T>
inline host_unique_ptr<T> get_host_unique_ptr(const std::size_t size){
	T* ptr;
	MTK_CUDA_CHECK_ERROR(cudaMallocHost((void**)&ptr, sizeof(T) * size));
	return std::unique_ptr<T, host_deleter<T>>{ptr};
}

// copy
template <class T>
inline void copy(T* const dst, const T* const src, const std::size_t size){
	MTK_CUDA_CHECK_ERROR(cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault));
}

} // memory
} // cuda
} // mtk

#endif // __CUTF_MEMORY_CUH__
