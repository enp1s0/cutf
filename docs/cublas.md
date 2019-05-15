# cuBLAS Functions
Not template but overload

## Example
```cpp
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
constexpr std::size_t N = 1<<10;

using T = half;
int main(){
	auto cublas_handle = cutf::cublas::get_ublas_unique_ptr();
	auto A = cutf::memory::get_device_unique_ptr<T>(N * N);
	auto B = cutf::memory::get_device_unique_ptr<T>(N * N);
	auto C = cutf::memory::get_device_unique_ptr<T>(N * N);
	T alpha = cutf::type::cast<T>(1.0f);
	T beta = cutf::type::cast<T>(1.0f);

	const auto status = cutf::cublas::gemm(*cublas_handle.get(),
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, N, N,
			&alpha,
			A.get(), N,
			B.get(), N,
			&beta,
			C.get(), N);

}


```

## Implemented operations
| Operation | cutf::cublas:: | half | float | double | cuComplex | cuDoubleComplex |
|:----------|:---------------|:-----|:------|:-------|:----------|:----------------|
|amax|iamax||S|D|C|Z|
|amin|iamin||S|D|C|Z|
|asum|asum||S|D|C|Z|
|axpy|axpy||S|D|C|Z|
|copy|copy||S|D|C|Z|
|dot|dot||S|D|||
|dotc|dotc||||C|Z|
|dotu|dotu||||C|Z|
|nrm2|nrm2||S|D|Sc|Dz|
|rot|rot||S|D|C/Cs|Z/Zd|
|rotg|rotg||S|D|C|Z|
|rotm|rotm||S|D|||
|rotmg|rotmg||S|D|||
|scal|scal||S|D|C/Cs|Z/Zd|
|swap|swap||S|D|C|Z|
|gbmv|gbmv||S|D|C|Z|
|gemv|gemv||S|D|C|Z|
|ger|ger||S|D|||
|gerc|gerc||||C|Z|
|geru|geru||||C|Z|
|sbmv|sbmv||S|D|||
|spmv|spmv||S|D|||
|spr|spr||S|D|||
|spr2|spr2||S|D|||
|symv|symv||S|D|C|Z|
|syr|syr||S|D|C|Z|
|syr2|syr2||S|D|C|Z|
|tbmv|tbmv||S|D|C|Z|
|tbsv|tbsv||S|D|C|Z|
|tpmv|tpmv||S|D|C|Z|
|tpsv|tpsv||S|D|C|Z|
|trmv|trmv||S|D|C|Z|
|trsv|trsv||S|D|C|Z|
|hemv|hemv||||C|Z|
|hbmv|hbmv||||C|Z|
|hpmv|hpmv||||C|Z|
|her|her||||C|Z|
|her2|her2||||C|Z|
|hpr|hpr||||C|Z|
|hpr2|hpr2||||C|Z|
|gemm|gemm|H|S|D|C|Z|
|gemm3m|gemm3m||S|D|C|Z|
|gemmBatched|gemm_batched|H|S|D|C|Z|
|gemmStridedBatched|gemm_strided_batched|H|S|D|C|Z|
|gemm3mStridedBatched|gemm3m_strided_batched||S|D|C|Z|
|symm|symm||S|D|C|Z|
|syrk|syrk||S|D|C|Z|
|syr2k|syr2k||S|D|C|Z|
|syrkx|syrkx||S|D|C|Z|
|trmm|trmm||S|D|C|Z|
|trsm|trsm||S|D|C|Z|
|trsmBatched|trsm_batched||S|D|C|Z|
|hemm|hemm||||C|Z|
|herk|herk||||C|Z|
|her2k|her2k||||C|Z|
|herkx|herkx||||C|Z|
|geam|geam||S|D|C|Z|
|dgmm|dgmm||S|D|C|Z|
|getrfBatched|getrf_batched||S|D|C|Z|
|getrsBatched|getrs_batched||S|D|C|Z|
|getriBatched|getri_batched||S|D|C|Z|
|matinvBatched|matinv_batched||S|D|C|Z|
|geqrfBatched|geqrf_batched||S|D|C|Z|
|gelsBatched|gels_batched||S|D|C|Z|
|tpttr|tpttr||S|D|C|Z|
|trttp|trttp||S|D|C|Z|

## Reference
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
