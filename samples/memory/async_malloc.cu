#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/stream.hpp>

constexpr std::size_t N = 1lu << 10;

int main() {
	auto cuda_stream = cutf::stream::get_stream_unique_ptr();

	auto a_ptr = cutf::memory::malloc_async<float>(N, *cuda_stream.get());

	CUTF_CHECK_ERROR(cutf::memory::free_async(a_ptr, *cuda_stream.get()));
}
