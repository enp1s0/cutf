#include <cutf/memory.hpp>

int main() {
	constexpr unsigned N = 1u << 20;

	// Standar device memory allocation
	auto device_ptr = cutf::memory::malloc<float>(N);
	cutf::memory::free(device_ptr);

	// Managed memory allocation
	auto managed_ptr = cutf::memory::malloc_managed<float>(N);
	cutf::memory::free(managed_ptr);

	// Host memory allocation
	auto host_ptr = cutf::memory::malloc_host<float>(N);
	cutf::memory::free_host(host_ptr);

	// Async device memory allocation
	cudaStream_t cuda_stream;
	CUTF_CHECK_ERROR(cudaStreamCreate(&cuda_stream));

	auto async_device_ptr = cutf::memory::malloc_async<float>(N, cuda_stream);
	cutf::memory::free_async(async_device_ptr, cuda_stream);

	CUTF_CHECK_ERROR(cudaStreamDestroy(cuda_stream));
}
