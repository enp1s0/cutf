#include <iostream>
#include <cutf/debug/time_breakdown.hpp>

constexpr unsigned N = 1u << 20;
constexpr unsigned block_size = 1u << 8;

__global__ void plus_one_kernel(
		float* const ptr,
		const unsigned N
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}

	ptr[tid] += 1.f;
}

int main() {
	cudaStream_t cuda_stream;
	cudaStreamCreate(&cuda_stream);

	cutf::debug::time_breakdown::profiler profiler(cuda_stream);

	float *da;
	profiler.start_timer_sync("cudaMalloc");
	cudaMalloc(reinterpret_cast<float**>(&da), sizeof(float));
	profiler.stop_timer_sync("cudaMalloc");

	profiler.start_timer_sync("kernel");
	plus_one_kernel<<<(N + block_size - 1) / block_size, block_size>>>(da, N);
	profiler.stop_timer_sync("kernel");

	profiler.print_result(stdout);

	cudaStreamDestroy(cuda_stream);
}
