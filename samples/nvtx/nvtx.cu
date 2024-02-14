#include <cutf/memory.hpp>
#include <cutf/nvtx.hpp>

constexpr std::size_t N = 1u << 20;

int main() {
	cutf::nvtx::range_push("TEST_RANGE_1");
	{
		auto d_mem = cutf::memory::get_device_unique_ptr<double>(N);
		cutf::nvtx::range_push("TEST_RANGE_2");
		CUTF_CHECK_ERROR(cudaMemset(d_mem.get(), 0xff, sizeof(double) * N));
		cutf::nvtx::range_pop();
	}
	cutf::nvtx::range_pop();
}
