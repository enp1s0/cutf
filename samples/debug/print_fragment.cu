#include <cutf/type.hpp>
#include <cutf/debug/fragment.hpp>
#include <cutf/debug/matrix.hpp>

__global__ void print_fragment_kernel() {
	constexpr unsigned N = 16;
	__shared__ half mat[N * N];

	const auto lane_id = cutf::thread::get_lane_id();

	for (unsigned i = 0; i < N * N; i += warpSize) {
		mat[i + lane_id] = cutf::type::cast<half>(static_cast<float>(i + lane_id));
	}

	__syncthreads();
	if (lane_id == 0) {
		cutf::debug::print::print_matrix(mat, N, N, N, "mat");
	}

	__syncthreads();
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> fragment;
	nvcuda::wmma::load_matrix_sync(fragment, mat, N);

	__syncthreads();
	cutf::debug::print::print_fragment(fragment, "frag");
}

int main() {
	print_fragment_kernel<<<1, 32>>>();
	cudaDeviceSynchronize();
}
