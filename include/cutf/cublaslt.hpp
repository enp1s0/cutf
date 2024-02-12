#ifndef __CUTF_CUBLASLT_CUH__
#define __CUTF_CUBLASLT_CUH__
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <sstream>
#include <memory>
#include "error.hpp"
#include "cublas.hpp"
#include "type.hpp"

namespace cutf {
namespace cublaslt {
namespace detail {
struct cublaslt_deleter{
	void operator()(cublasLtHandle_t* handle){
		CUTF_CHECK_ERROR(cublasLtDestroy(*handle));
		delete handle;
	}
};

struct matrix_layout_deleter{
	void operator()(cublasLtMatrixLayout_t* layout){
		CUTF_CHECK_ERROR(cublasLtMatrixLayoutDestroy(*layout));
		delete layout;
	}
};

struct matmul_desc_deleter{
	void operator()(cublasLtMatmulDesc_t* desc){
		CUTF_CHECK_ERROR(cublasLtMatmulDescDestroy(*desc));
		delete desc;
	}
};

struct preference_deleter{
	void operator()(cublasLtMatmulPreference_t* preference){
		CUTF_CHECK_ERROR(cublasLtMatmulPreferenceDestroy(*preference));
		delete preference;
	}
};
} // namespace detail

using handle_uptr_t = std::unique_ptr<cublasLtHandle_t, detail::cublaslt_deleter>;
using matrix_layout_uptr_t = std::unique_ptr<cublasLtMatrixLayout_t, detail::matrix_layout_deleter>;
using matmul_desc_uptr_t = std::unique_ptr<cublasLtMatmulDesc_t, detail::matmul_desc_deleter>;
using preference_uptr_t = std::unique_ptr<cublasLtMatmulPreference_t, detail::preference_deleter>;

inline handle_uptr_t create_handle_unique_ptr() {
	cublasLtHandle_t *handle = new cublasLtHandle_t;
	CUTF_CHECK_ERROR(cublasLtCreate(handle));
	return handle_uptr_t{handle};
}

template <class T>
inline matrix_layout_uptr_t create_matrix_layout_uptr(
    const std::size_t M,
    const std::size_t N,
    const std::size_t ld,
    const T* const ref_ptr = nullptr
    ) {
  cublasLtMatrixLayout_t *layout = new cublasLtMatrixLayout_t;
  CUTF_CHECK_ERROR(cublasLtMatrixLayoutCreate(
          layout,
          cutf::type::get_data_type<T>(),
          M, N, ld));
  return matrix_layout_uptr_t{layout};
}

inline matmul_desc_uptr_t create_matmul_desc_unique_ptr(
    const cublasComputeType_t compute_type,
    const cudaDataType_t scale_type
    ) {
	cublasLtMatmulDesc_t* desc = new cublasLtMatmulDesc_t;
	CUTF_CHECK_ERROR(cublasLtMatmulDescCreate(desc, compute_type, scale_type));
	return matmul_desc_uptr_t{desc};
}

inline preference_uptr_t create_preference_unique_ptr() {
	cublasLtMatmulPreference_t *preference = new cublasLtMatmulPreference_t;
	CUTF_CHECK_ERROR(cublasLtMatmulPreferenceCreate(preference));
	return preference_uptr_t{preference};
}
} // namespace cublaslt
} // namespace cutf
#endif // __CUTF_CUBLASLT_CUH__
