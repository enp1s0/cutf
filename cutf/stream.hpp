#ifndef __CUTF_STREAM_HPP__
#define __CUTF_STREAM_HPP__
#include <memory>
#include <cuda_runtime.h>
#include "cuda.hpp"

namespace cutf{
namespace stream{
struct stream_deleter{
	void operator()(cudaStream_t *stream){
		cudaStreamDestroy(*stream);
		delete stream;
	}
};

using stream_unique_ptr = std::unique_ptr<cudaStream_t, stream_deleter>;

inline stream_unique_ptr get_stream_unique_ptr(const int device_id = 0){
	cutf::error::check(cudaSetDevice(device_id), __FILE__, __LINE__, __func__, "@ Creating stream for device " + std::to_string(device_id));
	stream_unique_ptr stream_unique_ptr(new cudaStream_t);
	cutf::error::check(cudaStreamCreate(stream_unique_ptr.get()), __FILE__, __LINE__, __func__, "@ Creating stream for device " + std::to_string(device_id));
	cutf::error::check(cudaSetDevice(0), __FILE__, __LINE__, __func__, "@ Creating stream for device " + std::to_string(device_id));
	return stream_unique_ptr;
}
} // stream
} // cutf

#endif // __CUTF_STREAM_HPP__
