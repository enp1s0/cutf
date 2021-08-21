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

inline stream_unique_ptr get_stream_unique_ptr(){
	stream_unique_ptr stream(new cudaStream_t);
	cutf::error::check(cudaStreamCreate(stream.get()), __FILE__, __LINE__, __func__, "@ Creating stream");
	return stream;
}
} // stream
} // cutf

#endif // __CUTF_STREAM_HPP__
