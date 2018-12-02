#include <random>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

#define CUBLAS_ERROR_HANDLE(status) cutf::cublas::error::check(status, __FILE__, __LINE__, __func__)

using compute_t = float;
const std::size_t N = 1 << 10;

int main(){
	auto hA = cutf::cuda::memory::get_host_unique_ptr<compute_t>(N * N);
	auto hB = cutf::cuda::memory::get_host_unique_ptr<compute_t>(N * N);
	auto hC = cutf::cuda::memory::get_host_unique_ptr<compute_t>(N * N);
	auto dA = cutf::cuda::memory::get_device_unique_ptr<compute_t>(N * N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<compute_t>(N * N);
	auto dC = cutf::cuda::memory::get_device_unique_ptr<compute_t>(N * N);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < N * N; i++){
		hA.get()[i] = dist(mt);
		hB.get()[i] = dist(mt);
	}

	cutf::cuda::memory::copy(dA.get(), hA.get(), N * N);
	cutf::cuda::memory::copy(dB.get(), hB.get(), N * N);

	auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
	compute_t alpha = cutf::cuda::type::cast<compute_t>(1.0f);
	compute_t beta = cutf::cuda::type::cast<compute_t>(1.0f);

	CUBLAS_ERROR_HANDLE(cutf::cublas::gemm(
				*cublas_handle.get(),
				CUBLAS_OP_N, CUBLAS_OP_N,
				N, N, N,
				&alpha,
				dA.get(), N,
				dB.get(), N,
				&beta,
				dC.get(), N
				));

	cutf::cuda::memory::copy(hC.get(), dC.get(), N * N);
}
