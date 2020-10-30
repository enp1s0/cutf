#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/debug/type.hpp>
#include <cutf/experimental/exponent.hpp>

constexpr std::size_t N = 1lu << 20;
constexpr std::size_t block_size = 1lu << 8;

template <class T, int min_exponent>
__global__ void force_underflow_kernel(T* const dst_ptr, const T* const src_ptr, const std::size_t N) {
	const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid >= N) return;

	dst_ptr[tid] = cutf::experimental::exponent::min_exponent(src_ptr[tid], min_exponent);
}

template <class T, int min_exponent>
void test_force_underflow() {
	const double range = std::pow<double>(2.0, min_exponent + 1);
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dist(-range, range);
	auto h_org = cutf::memory::get_host_unique_ptr<T>(N);
	auto h_aft = cutf::memory::get_host_unique_ptr<T>(N);

	for (std::size_t i = 0; i < N; i++) {
		h_org.get()[i] = cutf::type::cast<T>(dist(mt));
	}
	
	force_underflow_kernel<T, min_exponent><<<(N + block_size - 1) / block_size, block_size>>>(h_aft.get(), h_org.get(), N);
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());

	std::size_t error_count = 0;
	for (std::size_t i = 0; i < N; i++) {
		const auto org_bs = cutf::experimental::fp::reinterpret_as_uint(h_org.get());
		const auto aft_bs = cutf::experimental::fp::reinterpret_as_uint(h_aft.get());

		if (org_bs != aft_bs) {
			// expected:
			// - aft_bs == 0
			// - exponent of `org_bs` is smaller than `min_exponent`
			const auto exponent_org = (org_bs << 1) >> (1 + cutf::experimental::fp::get_mantissa_size<T>());
			if (! (((aft_bs << 1) >> 1) == 0 && (static_cast<int>(exponent_org) - static_cast<int>(cutf::experimental::fp::get_bias<T>()) < min_exponent))) {
				error_count++;
			}
		}
	}

	std::printf("type = %10s, min_exponent = %5d [%10lu / %10lu (%3.2f%%)]\n", cutf::debug::type::get_type_name<T>(), min_exponent, error_count, N, static_cast<double>(error_count) / N * 100.0);
}

int main() {
	test_force_underflow<half   , -1  >();
	test_force_underflow<half   , -4  >();
	test_force_underflow<half   , -10 >();

	test_force_underflow<float , -10 >();
	test_force_underflow<float , -20 >();
	test_force_underflow<float , -100>();

	test_force_underflow<double, -10 >();
	test_force_underflow<double, -20 >();
	test_force_underflow<double, -100>();
}
