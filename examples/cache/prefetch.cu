#include <iostream>
#include <chrono>
#include <type_traits>
#include <cutf/memory.hpp>
#include <cutf/cache.hpp>

const std::size_t N = 1lu << 30;
const std::size_t C = 1lu << 6;

template <class prefetch>
__global__ void prefetch_test_kernel(
		int* const ptr,
		const std::size_t length
		) {
	const auto i_stride = blockDim.x * gridDim.x;
	std::size_t i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= length) {
		return;
	}

	for (; i < length - i_stride; i += i_stride) {
		ptr[i] += i;

		if constexpr (!std::is_same<prefetch, void>::value) {
			cutf::cache::prefetch<prefetch, cutf::cache::global>(ptr + i + i_stride);
		}
	}

	ptr[i] += i;
}

template <class prefetch>
void prefetch_test(
		int* const ptr,
		const std::size_t length
		) {
	const std::size_t block_size = 256;
	const std::size_t grid_size = 256;

	prefetch_test_kernel<prefetch><<<grid_size, block_size>>>(
			ptr,
			length
			);
}

template <class prefetch_mode>
std::string get_name_str();
template <> std::string get_name_str<cutf::cache::L1             >() {return "L1";}
template <> std::string get_name_str<cutf::cache::L2             >() {return "L2";}
template <> std::string get_name_str<cutf::cache::L2_evict_last  >() {return "L2_evict_last";}
template <> std::string get_name_str<cutf::cache::L2_evict_normal>() {return "L2_evict_normal";}
template <> std::string get_name_str<void                        >() {return "void";}

template <class prefetch>
void eval(const std::size_t N) {
	auto mem_array = cutf::memory::get_device_unique_ptr<int>(N);
	CUTF_CHECK_ERROR(cudaMemset(mem_array.get(), 0, sizeof(int) * N));

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto start_clock = std::chrono::system_clock::now();

	for (unsigned i = 0; i < C; i++) {
		prefetch_test<prefetch>(mem_array.get(), N);
	}

	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	const auto end_clock = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;
	std::printf(
			"Prefetch mode = %s, Bandwidth = %e [TB/s]\n",
			get_name_str<prefetch>().c_str(),
			1. * N * sizeof(int) / elapsed_time * 1e-12 * C * 2
			);
}

int main() {
	eval<void                        >(N);
	eval<cutf::cache::L1             >(N);
	eval<cutf::cache::L2             >(N);
	eval<cutf::cache::L2_evict_normal>(N);
	eval<cutf::cache::L2_evict_last  >(N);
}
