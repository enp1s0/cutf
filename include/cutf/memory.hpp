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
		CUTF_CHECK_ERROR(cudaFree( ptr ));
	}
};
template <class T>
class host_deleter{
public:
	void operator()(T* ptr){
		CUTF_CHECK_ERROR(cudaFreeHost( ptr ));
	}
};

// unique pointer type for c++ 11
template <class T>
using device_unique_ptr = std::unique_ptr<T, device_deleter<T>>;
template <class T>
using host_unique_ptr = std::unique_ptr<T, host_deleter<T>>;

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

// Standard malloc / free
template <class T>
inline T* malloc(const std::size_t count) {
	T* ptr;
	CUTF_CHECK_ERROR_M(cudaMalloc((void**)&ptr, sizeof(T) * count), "Failed to allocate " + std::to_string(count * sizeof(T)) + " Bytes of device memory");
	return ptr;
}

template <class T>
inline void free(T* const ptr) {
	CUTF_CHECK_ERROR(cudaFree(ptr));
}

// asyn malloc/free
// NOTE: These functions are only available in CUDA >= 11.2
template <class T>
inline T* malloc_async(const std::size_t count, const cudaStream_t stream) {
#ifndef CUTF_DISABLE_MALLOC_ASYNC
	T* ptr;
	CUTF_CHECK_ERROR_M(cudaMallocAsync((void**)&ptr, sizeof(T) * count, stream), "Failed to allocate " + std::to_string(count * sizeof(T)) + " Bytes of device memory");
	return ptr;
#else
	CUTF_CHECK_ERROR(cudaStreamSynchronize(stream));
	return cutf::memory::malloc<T>(count);
#endif
}

template <class T>
inline void free_async(T* const ptr, const cudaStream_t stream) {
#ifndef CUTF_DISABLE_MALLOC_ASYNC
	CUTF_CHECK_ERROR(cudaFreeAsync(ptr, stream));
#else
	CUTF_CHECK_ERROR(cudaStreamSynchronize(stream));
	return cutf::memory::free(ptr);
#endif
}

// Managed
// Managed memory should be released with cudaFree
template <class T>
inline T* malloc_managed(const std::size_t count) {
	T* ptr;
	CUTF_CHECK_ERROR_M(cudaMallocManaged((void**)&ptr, sizeof(T) * count), "Failed to allocate " + std::to_string(count * sizeof(T)) + " Bytes of managed memory");
	return ptr;
}

// Host
template <class T>
inline T* malloc_host(const std::size_t count) {
	T* ptr;
	CUTF_CHECK_ERROR_M(cudaMallocHost((void**)&ptr, sizeof(T) * count), "Failed to allocate " + std::to_string(count * sizeof(T)) + " Bytes of host memory");
	return ptr;
}

template <class T>
inline void free_host(T* const ptr) {
	CUTF_CHECK_ERROR(cudaFreeHost(ptr));
}

// -----------------------------------------------
// Unique ptrs
// -----------------------------------------------
// allocater
template <class T>
inline device_unique_ptr<T> get_device_unique_ptr(const std::size_t size){
	T* const ptr = malloc<T>(size);
	return std::unique_ptr<T, device_deleter<T>>{ptr};
}

template <class T>
inline device_unique_ptr<T> get_managed_unique_ptr(const std::size_t size){
	T* const ptr = malloc_managed<T>(size);
	return std::unique_ptr<T, device_deleter<T>>{ptr};
}

template <class T>
inline host_unique_ptr<T> get_host_unique_ptr(const std::size_t size){
	T* const ptr = malloc_host<T>(size);
	return std::unique_ptr<T, host_deleter<T>>{ptr};
}

} // memory
} // cutf

#endif // __CUTF_MEMORY_CUH__
