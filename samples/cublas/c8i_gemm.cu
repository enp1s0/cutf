#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>

#define CUBLAS_ERROR_HANDLE(status) cutf::error::check(status, __FILE__, __LINE__, __func__)

constexpr std::size_t N = 1 << 13;

using compute_t = cuComplex;
using ab_t= short;
using c_t= cuComplex;

int main(){
	auto dA = cutf::memory::get_device_unique_ptr<ab_t>(N * N);
	auto dB = cutf::memory::get_device_unique_ptr<ab_t>(N * N);
	auto dC = cutf::memory::get_device_unique_ptr<c_t>(N * N);

	compute_t alpha, beta;

	auto cublas = cutf::cublas::get_cublas_unique_ptr();

	CUBLAS_ERROR_HANDLE(cublasGemmEx(
				*cublas.get(),
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				N, N, N,
				&alpha,
				dA.get(), cutf::type::get_data_type<ab_t>(), N,
				dB.get(), cutf::type::get_data_type<ab_t>(), N,
				&beta,
				dC.get(), cutf::type::get_data_type<c_t>(), N,
				cutf::type::get_data_type<compute_t>(),
				CUBLAS_GEMM_DEFAULT
				));
}
