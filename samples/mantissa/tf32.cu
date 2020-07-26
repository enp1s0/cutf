#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/debug/matrix.hpp>
constexpr unsigned warp_size = 32;

__global__ void m16n16k16_tf32(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	constexpr unsigned N = 16;
	const unsigned lane_id = threadIdx.x & 0x1f;

	const auto m = lane_id & 0xf;
	const auto n_offset = lane_id / N;
	for (unsigned i = 0; i < N; i+= warp_size / N) {
		const auto n = i + n_offset;
		float sum = 0.0f;
		for (unsigned k = 0; k < N; k++) {
			sum += cutf::type::cast<nvcuda::wmma::precision::tf32>(a_ptr[m + k * N]) * cutf::type::cast<nvcuda::wmma::precision::tf32>(b_ptr[k + n * N]);
		}
		c_ptr[m + n * N] += sum;
	}
}

__global__ void m16n16k16_fp32(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
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

double get_max_error(const float* const fp32_ptr, const float* const tf32_ptr, const unsigned m, const unsigned n) {
	double max_error = 0.0;
	for (unsigned i = 0; i < m; i++) {
		for (unsigned j = 0; j < n; j++) {
			max_error = std::max(std::abs(static_cast<double>(fp32_ptr[i * n + j]) - static_cast<double>(tf32_ptr[i * n + j])), max_error);
		}
	}
	return max_error;
}

int main() {
	constexpr unsigned N = 16;

	auto A = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto B = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto C_tf32 = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto C_fp32 = cutf::memory::get_host_unique_ptr<float>(N * N);

	std::mt19937 mt(std::random_device{}());
	float max_range = 1.0f;
	std::uniform_real_distribution<float> dist(-max_range, max_range);

	for (unsigned i = 0; i < N * N; i++) {
		A.get()[i] = dist(mt);
		B.get()[i] = dist(mt);
		C_tf32.get()[i] = 0.0f;
		C_fp32.get()[i] = 0.0f;
	}

	m16n16k16_tf32<<<1, warp_size>>>(C_tf32.get(), A.get(), B.get());
	m16n16k16_fp32<<<1, warp_size>>>(C_fp32.get(), A.get(), B.get());

	cudaDeviceSynchronize();

	std::printf("max_error = %e\n", get_max_error(C_fp32.get(), C_tf32.get(), N, N));
}
