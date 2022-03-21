#include <iostream>
#include <cutf/debug/time_breakdown.hpp>

constexpr unsigned N = 1u << 20;
constexpr unsigned block_size = 1u << 8;

__global__ void init_kernel(
		float* const ptr,
		const unsigned N
		) {
	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= N) {
		return;
	}

	ptr[tid] = 1.f;
}


__global__ void add_1_kernel(
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
	profiler.measure(
			"cudaMalloc",
			[&]() {
				cudaMalloc(reinterpret_cast<float**>(&da), sizeof(float) * N);
			});

	profiler.start_timer_sync("init_kernel");
	init_kernel<<<(N + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(da, N);
	profiler.stop_timer_sync("init_kernel");

	for (unsigned i = 0; i < 100; i++) {
		profiler.start_timer_sync("add_1_kernel");
		add_1_kernel<<<(N + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(da, N);
		profiler.stop_timer_sync("add_1_kernel");
	}

	profiler.disable_measurement();
	profiler.start_timer_sync("should_not_be_measured");
	add_1_kernel<<<(N + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(da, N);
	profiler.stop_timer_sync("should_not_be_measured");
	profiler.enable_measurement();

	float ha[N];
	profiler.start_timer_sync("cudaMemcpy");
	cudaMemcpy(ha, da, sizeof(float) * N, cudaMemcpyDefault);
	profiler.stop_timer_sync("cudaMemcpy");

	profiler.print_result(stdout);
	profiler.print_result_csv(stdout);

	cudaStreamDestroy(cuda_stream);
}
