#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/experimental/mantissa.hpp>
#include <cutf/type.hpp>
#include <cutf/debug/matrix.hpp>
constexpr unsigned warp_size = 32;

__global__ void m16n16k16_cut(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	constexpr unsigned N = 16;
	const unsigned lane_id = threadIdx.x & 0x1f;

	const auto m = lane_id & 0xf;
	const auto n_offset = lane_id / N;
	for (unsigned i = 0; i < N; i+= warp_size / N) {
		const auto n = i + n_offset;
		float sum = 0.0f;
		for (unsigned k = 0; k < N; k++) {
			sum += cutf::experimental::mantissa::cut_mantissa<10, cutf::rounding::rr>(a_ptr[m + k * N]) * cutf::experimental::mantissa::cut_mantissa<10, cutf::rounding::rr>(b_ptr[k + n * N]);
		}
		c_ptr[m + n * N] += sum;
	}
}

__global__ void m16n16k16_half(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	constexpr unsigned N = 16;
	const unsigned lane_id = threadIdx.x & 0x1f;

	const auto m = lane_id & 0xf;
	const auto n_offset = lane_id / N;
	for (unsigned i = 0; i < N; i+= warp_size / N) {
		const auto n = i + n_offset;
		float sum = 0.0f;
		for (unsigned k = 0; k < N; k++) {
			sum += cutf::type::cast<float>(cutf::type::cast<half>(a_ptr[m + k * N])) * cutf::type::cast<float>(cutf::type::cast<half>(b_ptr[k + n * N]));
		}
		c_ptr[m + n * N] += sum;
	}
}

__global__ void m16n16k16_base(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	constexpr unsigned N = 16;
	const unsigned lane_id = threadIdx.x & 0x1f;

	const auto m = lane_id & 0xf;
	const auto n_offset = lane_id / N;
	for (unsigned i = 0; i < N; i+= warp_size / N) {
		const auto n = i + n_offset;
		float sum = 0.0f;
		for (unsigned k = 0; k < N; k++) {
			sum += a_ptr[m + k * N] * b_ptr[k + n * N];
		}
		c_ptr[m + n * N] += sum;
	}
}

double get_max_error(const float* const base_ptr, const float* const cut_ptr, const unsigned m, const unsigned n) {
	double max_error = 0.0;
	for (unsigned i = 0; i < m; i++) {
		for (unsigned j = 0; j < n; j++) {
			max_error = std::max(std::abs(static_cast<double>(base_ptr[i * n + j]) - cut_ptr[i * n + j]), max_error);
		}
	}
	return max_error;
}

int main() {
	constexpr unsigned N = 16;

	auto A = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto B = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto C_cut = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto C_half = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto C_base = cutf::memory::get_host_unique_ptr<float>(N * N);

	std::mt19937 mt(std::random_device{}());
	float max_range = 1.0f;
	std::uniform_real_distribution<float> dist(-max_range, max_range);

	for (unsigned i = 0; i < N * N; i++) {
		A.get()[i] = dist(mt);
		B.get()[i] = dist(mt);
		C_cut.get()[i] = 0.0f;
		C_half.get()[i] = 0.0f;
		C_base.get()[i] = 0.0f;
	}

	m16n16k16_cut<<<1, warp_size>>>(C_cut.get(), A.get(), B.get());
	m16n16k16_half<<<1, warp_size>>>(C_half.get(), A.get(), B.get());
	m16n16k16_base<<<1, warp_size>>>(C_base.get(), A.get(), B.get());

	cudaDeviceSynchronize();

	std::printf("[cut ] max_error = %e\n", get_max_error(C_base.get(), C_half.get(), N, N));
	std::printf("[half] max_error = %e\n", get_max_error(C_base.get(), C_cut.get(), N, N));
}
