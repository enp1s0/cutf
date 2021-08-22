#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/event.hpp>

constexpr std::size_t N = 1lu << 20;
constexpr std::size_t block_size = 256;

__global__ void vector_add(float* const dst, const float* const src_a, const float* const src_b) {
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	dst[tid] = src_a[tid] + src_b[tid];
}

int main() {
	auto dA = cutf::memory::get_device_unique_ptr<float>(N);
	auto dB = cutf::memory::get_device_unique_ptr<float>(N);
	auto dC = cutf::memory::get_device_unique_ptr<float>(N);

	auto event_start = cutf::event::get_event_unique_ptr();
	auto event_end = cutf::event::get_event_unique_ptr();

	cudaEventRecord(*event_start.get());
	vector_add<<<N / block_size, block_size>>>(dC.get(), dA.get(), dB.get());
	cudaEventRecord(*event_end.get());
	cudaEventSynchronize(*event_end.get());

	const auto elapsed_time = cutf::event::get_elapsed_time(*event_start.get(), *event_end.get());
	std::printf("Elapsed time = %e [ms]\n", elapsed_time);
}
