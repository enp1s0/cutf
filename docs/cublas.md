# cuBLAS Functions
## Example
```cpp
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cublas.hpp>
constexpr std::size_t N = 1<<10;

using T = half;
int main(){
	cublasHandle_t cublas;
	cublasCreate( &cublas );
	auto A = mtk::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto B = mtk::cuda::memory::get_device_unique_ptr<T>(N * N);
	auto C = mtk::cuda::memory::get_device_unique_ptr<T>(N * N);
	T alpha = mtk::cuda::type::cast<T>(1.0f);
	T beta = mtk::cuda::type::cast<T>(1.0f);

	cutf::cublas::gemm(cublas,
			CUBLAS_OP_N, CUBLAS_OP_N,
			N, N, N,
			&alpha,
			A.get(), N,
			B.get(), N,
			&beta,
			C.get(), N);

	cublasDestroy( cublas );
}


```

## Implemented operations
| Operation | half | float | double | cuComplex | cuDoubleComplex |
|:----------|:-----|:------|:-------|:----------|:----------------|
|amax||S|D|C|Z|
|amin||S|D|C|Z|
|asum||S|D|C|Z|
|axpy||S|D|C|Z|
|copy||S|D|C|Z|
|dot||S|D|||
|dotc||||C|Z|
|dotu||||C|Z|
|nrm2||S|D|Sc|Dz|
|rot||S|D|C/Cs|Z/Zd|
|rotg||S|D|C|Z|
|rotm||S|D|||
|rotmg||S|D|||
|scal||S|D|C/Cs|Z/Zd|
|swap||S|D|C|Z|
|gbmv||S|D|C|Z|
|gemv||S|D|C|Z|
|ger||S|D|||
|gerc||||C|Z|
|geru||||C|Z|
|sbmv||S|D|||
|spmv||S|D|||
|spr||S|D|||
|spr2||S|D|||
|symv||S|D|C|Z|
|syr||S|D|C|Z|
|syr2||S|D|C|Z|
|tbmv||S|D|C|Z|
|tbsv||S|D|C|Z|
|tpmv||S|D|C|Z|
|tpsv||S|D|C|Z|
|trmv||S|D|C|Z|
|trsv||S|D|C|Z|
|hemv||||C|Z|
|hbmv||||C|Z|
|hpmv||||C|Z|
|her||||C|Z|
|her2||||C|Z|
|hpr||||C|Z|
|hpr2||||C|Z|
|gemm|H|S|D|C|Z|
|gemm3m||S|D|C|Z|
|gemmBatched|H|S|D|C|Z|
|gemmStridedBatched|H|S|D|C|Z|
|gemm3mStridedBatched||S|D|C|Z|
|symm||S|D|C|Z|
|syrk||S|D|C|Z|
|syr2k||S|D|C|Z|
|syrkx||S|D|C|Z|
|trmm||S|D|C|Z|
|trsm||S|D|C|Z|
|trsmBatched||S|D|C|Z|
|hemm||||C|Z|
|herk||||C|Z|
|her2k||||C|Z|
|herkx||||C|Z|
|geam||S|D|C|Z|
|dgmm||S|D|C|Z|
|getrfBatched||S|D|C|Z|
|getrsBatched||S|D|C|Z|
|getriBatched||S|D|C|Z|
|matinvBatched||S|D|C|Z|
|geqrfBatched||S|D|C|Z|
|gelsBatched||S|D|C|Z|
|tpttr||S|D|C|Z|
|trttp||S|D|C|Z|


