#include <iostream>
#include <random>
#include <cutf/memory.hpp>
#include <cutf/debug/tf32.hpp>
#include <cutf/debug/matrix.hpp>
constexpr unsigned warp_size = 32;

__global__ void m16n16k16(float* const c_ptr, const float* const a_ptr, const float* const b_ptr) {
	constexpr unsigned N = 16;
	const unsigned lane_id = threadIdx.x & 0x1f;

	const auto m = lane_id & 0xf;
	const auto n_offset = lane_id / N;
	for (unsigned i = 0; i < N; i+= warp_size / N) {
		const auto n = i + n_offset;
		float sum = 0.0f;
		for (unsigned k = 0; k < N; k++) {
			sum += cutf::debug::tf32::to_tf32(a_ptr[m + k * N]) * cutf::debug::tf32::to_tf32(b_ptr[k + n * N]);
		}
		c_ptr[m + n * N] += sum;
	}
}
int main() {
	constexpr unsigned N = 16;

	auto A = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto B = cutf::memory::get_host_unique_ptr<float>(N * N);
	auto C = cutf::memory::get_host_unique_ptr<float>(N * N);

	std::mt19937 mt(std::random_device{}());
	float max_range = 1.0f;
	std::uniform_real_distribution<float> dist(-max_range, max_range);

	for (unsigned i = 0; i < N * N; i++) {
		A.get()[i] = dist(mt);
		B.get()[i] = dist(mt);
		C.get()[i] = 0.0f;
	}

	m16n16k16<<<1, warp_size>>>(C.get(), A.get(), B.get());

	cutf::debug::matrix::print_matrix(C.get(), N, N, "C");
}
