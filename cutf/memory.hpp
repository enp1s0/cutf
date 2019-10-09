#ifndef __CUTF_MEMORY_CUH__
#define __CUTF_MEMORY_CUH__

#include <memory>
#include <cuda_runtime.h>
#include "cuda.hpp"
#include "error.hpp"


namespace cutf{
namespace memory{

// deleter
template <class T>
class device_deleter{
public:
	void operator()(T* ptr){
		CUTF_HANDLE_ERROR(cudaFree( ptr ));
	}
};
template <class T>
class host_deleter{
public:
	void operator()(T* ptr){
		CUTF_HANDLE_ERROR(cudaFreeHost( ptr ));
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
	CUTF_HANDLE_ERROR_M(cudaMalloc((void**)&ptr, sizeof(T) * size), "Failed to allocate " + std::to_string(size * sizeof(T)) + " Bytes of device memory");
	return std::unique_ptr<T, device_deleter<T>>{ptr};
}
template <class T>
inline host_unique_ptr<T> get_host_unique_ptr(const std::size_t size){
	T* ptr;
	CUTF_HANDLE_ERROR_M(cudaMallocHost((void**)&ptr, sizeof(T) * size), "Failed to allocate " + std::to_string(size * sizeof(T)) + " Bytes of host memory");
	return std::unique_ptr<T, host_deleter<T>>{ptr};
}

// copy
template <class T>
inline cudaError_t copy(T* const dst, const T* const src, const std::size_t size){
	return cudaMemcpy(dst, src, sizeof(T) * size, cudaMemcpyDefault);
}

// asynchronous copy
template <class T>
inline cudaError_t copy_async(T* const dst, const T* const src, const std::size_t size, cudaStream_t stream = 0){
	return cudaMemcpyAsync(dst, src, sizeof(T) * size, cudaMemcpyDefault, stream);
}

} // memory
} // cutf

#endif // __CUTF_MEMORY_CUH__
