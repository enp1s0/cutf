#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/debug/clock_breakdown.hpp>

constexpr std::size_t N = 256;

__global__ void kernel(float* const ptr) {
	CUTF_CLOCK_BREAKDOWN_INIT(2);
	CUTF_CLOCK_BREAKDOWN_RECORD(0);

	const auto tid = blockDim.x * blockIdx.x + threadIdx.x;

	const auto v = ptr[tid];
	ptr[tid] = v * v;

	CUTF_CLOCK_BREAKDOWN_RECORD(1);

	printf("%lld\n",
			CUTF_CLOCK_BREAKDOWN_DURATION(0, 1)
			);
}

int main() {
	auto ha = cutf::memory::get_host_unique_ptr<float>(N);
	kernel<<<1, N>>>(ha.get());
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
}
