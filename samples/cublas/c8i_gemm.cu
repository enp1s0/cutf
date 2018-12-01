#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>

#define CUBLAS_ERROR_HANDLE(status) cutf::cublas::error::check(status, __FILE__, __LINE__, __func__)

constexpr std::size_t N = 1 << 13;

using compute_t = cuComplex;
using ab_t= short;
using c_t= cuComplex;

int main(){
	auto dA = cutf::cuda::memory::get_device_unique_ptr<ab_t>(N * N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<ab_t>(N * N);
	auto dC = cutf::cuda::memory::get_device_unique_ptr<c_t>(N * N);

	compute_t alpha, beta;

	cublasHandle_t cublas_handle;
	CUBLAS_ERROR_HANDLE(cublasCreate(&cublas_handle));
	CUBLAS_ERROR_HANDLE(cublasGemmEx(
				cublas_handle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				N, N, N,
				&alpha,
				dA.get(), cutf::cuda::type::get_data_type<ab_t>(), N,
				dB.get(), cutf::cuda::type::get_data_type<ab_t>(), N,
				&beta,
				dC.get(), cutf::cuda::type::get_data_type<c_t>(), N,
				cutf::cuda::type::get_data_type<compute_t>(),
				CUBLAS_GEMM_DEFAULT
				));
}
