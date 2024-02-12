#include <random>
#include <cutf/cublaslt.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

using compute_t = float;
const std::size_t N = 1 << 10;

int main(){
	auto hA = cutf::memory::get_host_unique_ptr<compute_t>(N * N);
	auto hB = cutf::memory::get_host_unique_ptr<compute_t>(N * N);
	auto hC = cutf::memory::get_host_unique_ptr<compute_t>(N * N);
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(N * N);
	auto dB = cutf::memory::get_device_unique_ptr<compute_t>(N * N);
	auto dC = cutf::memory::get_device_unique_ptr<compute_t>(N * N);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < N * N; i++){
		hA.get()[i] = dist(mt);
		hB.get()[i] = dist(mt);
	}

	cutf::memory::copy(dA.get(), hA.get(), N * N);
	cutf::memory::copy(dB.get(), hB.get(), N * N);

	auto cublaslt_handle = cutf::cublaslt::create_handle_unique_ptr();
	compute_t alpha = cutf::type::cast<compute_t>(1.0f);
	compute_t beta = cutf::type::cast<compute_t>(1.0f);

  auto a_desc_uptr = cutf::cublaslt::create_matrix_layout_uptr(
      N, N, N, dA.get()
      );
  auto b_desc_uptr = cutf::cublaslt::create_matrix_layout_uptr(
      N, N, N, dB.get()
      );
  auto c_desc_uptr = cutf::cublaslt::create_matrix_layout_uptr(
      N, N, N, dC.get()
      );

  auto cublaslt_op_desc = cutf::cublaslt::create_matmul_desc_unique_ptr(
      CUBLAS_COMPUTE_32F,
      cutf::type::get_data_type<compute_t>()
      );

  auto cublaslt_preference_uptr = cutf::cublaslt::create_preference_unique_ptr();
  const std::size_t worksize = 4lu << 20;
  CUTF_CHECK_ERROR(cublasLtMatmulPreferenceSetAttribute(
          *cublaslt_preference_uptr.get(),
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &worksize,
          sizeof(worksize)
          ));

  int returned_results = 0;
  cublasLtMatmulHeuristicResult_t heuristic_result = {};
  CUTF_CHECK_ERROR(cublasLtMatmulAlgoGetHeuristic(
          *cublaslt_handle.get(),
          *cublaslt_op_desc.get(),
          *a_desc_uptr.get(),
          *b_desc_uptr.get(),
          *c_desc_uptr.get(),
          *c_desc_uptr.get(),
          *cublaslt_preference_uptr.get(),
          1,
          &heuristic_result,
          &returned_results
          ));

  auto workspace_uptr = cutf::memory::get_device_unique_ptr<std::uint8_t>(worksize);

  CUTF_CHECK_ERROR(cublasLtMatmul(
          *cublaslt_handle.get(),
          *cublaslt_op_desc.get(),
          &alpha,
          a_desc_uptr.get(), *a_desc_uptr.get(),
          b_desc_uptr.get(), *b_desc_uptr.get(),
          &beta,
          c_desc_uptr.get(), *c_desc_uptr.get(),
          c_desc_uptr.get(), *c_desc_uptr.get(),
          &heuristic_result.algo,
          workspace_uptr.get(),
          worksize,
          0
          ));

	cutf::memory::copy(hC.get(), dC.get(), N * N);
}
