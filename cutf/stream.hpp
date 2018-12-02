#ifndef __CUTF_STREAM_HPP__
#define __CUTF_STREAM_HPP__
#include <memory>
#include <cuda_runtime.h>
#include "error.hpp"

namespace cutf{
namespace cuda{
namespace stream{
struct stream_deleter{
	void operator()(cudaStream_t *stream){
		cudaStreamDestroy(*stream);
		delete stream;
	}
};

inline std::unique_ptr<cudaStream_t, stream_deleter> get_stream_unique_ptr(const int device_id = 0){
	cutf::cuda::error::check(cudaSetDevice(device_id), __FILE__, __LINE__, __func__, "@ Creating stream for device " + std::to_string(device_id));
	std::unique_ptr<cudaStream_t, stream_deleter> stream_unique_ptr(new cudaStream_t);
	cutf::cuda::error::check(cudaStreamCreate(stream_unique_ptr.get()), __FILE__, __LINE__, __func__, "@ Creating stream for device " + std::to_string(device_id));
	cutf::cuda::error::check(cudaSetDevice(0), __FILE__, __LINE__, __func__, "@ Creating stream for device " + std::to_string(device_id));
	return stream_unique_ptr;
}
} // stream
} // cuda
} // cutf

#endif // __CUTF_STREAM_HPP__
