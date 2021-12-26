#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/debug/matrix.hpp>

constexpr unsigned N = 4;

int main() {
	float h_mat[N * N];
	for (unsigned i = 0; i < N * N; i++) {
		h_mat[i] = i;
	}
	std::printf("# print_matrix\n");
	cutf::debug::print::print_matrix(h_mat, N, N);
	cutf::debug::print::print_numpy_matrix(h_mat, N, N);
	cutf::debug::print::print_matrix(h_mat, N, N, N);
	cutf::debug::print::print_numpy_matrix(h_mat, N, N, N);

	auto d_uptr = cutf::memory::get_device_unique_ptr<float>(N * N);
	cutf::memory::copy(d_uptr.get(), h_mat, N * N);

	std::printf("# print_matrix_from_host\n");
	cutf::debug::print::print_matrix_from_host(d_uptr.get(), N, N);
	cutf::debug::print::print_numpy_matrix_from_host(d_uptr.get(), N, N);
	cutf::debug::print::print_matrix_from_host(d_uptr.get(), N, N, N);
	cutf::debug::print::print_numpy_matrix_from_host(d_uptr.get(), N, N, N);
}
