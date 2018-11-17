# cutf - CUDA Template Functions
The library of the CUDA/C++ Otaku, by the CUDA/C++ Otaku(?), for the CUDA/C++ Otaku shall not perish from the earth.

## Example
```cpp
#include "cutf/math.cuh"
#include "cutf/type.cuh"
#include "cutf/memory.cuh"
constexpr float PI = 3.f;
constexpr std::size_t N = 15;

template <class T, int N>
__global__ void kernel_example(T* const output, const T* const input){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= N ) return;

	output[tid] = mtk::cuda::math::sin( __ldg(input + tid) * mtk::cuda::type::cast<T>(PI) );
}

using T = float;
int main(){
	auto in = mtk::cuda::memory::get_device_unique_ptr<T>(N);
	auto out = mtk::cuda::memory::get_device_unique_ptr<T>(N);
	auto h_out = mtk::cuda::memory::get_host_unique_ptr<T>(N);

	kernel_example<T, N><<<(N+15)/16,16>>>(out.get(), in.get());

	mtk::cuda::memory::copy(h_out.get(), out.get(), N);
}
```

## Functions
- math
	- ceil
	- cos
	- exp
	- exp10
	- exp2
	- floor
	- log
	- log10
	- log2
	- rcp
	- rint
	- rsqrt
	- sin
	- sqrt
	- trunc

- type
	- cast
	- rounding cast

- memory
	- get\_device\_unique\_ptr
	- get\_host\_unique\_ptr
	- copy

## cuBLAS
| operation | half | float | double | cuComplex | cuDoubleComplex |
|:----------|:-----|:------|:-------|:----------|:----------------|
|amax||S|D|C|Z|
|amin||S|D|C|Z|
|asum||S|D|C|Z|
|axpy||S|D|C|Z|
|copy||S|D|C|Z|
|dot||S|D|C - u/c|Z - u/c|
|nrm2||S|D|Sc|Dz|
|rot||S|D|C/Cs|Z/Zd|
|rotg||S|D|C|Z|
|rotm||S|D|||
|rotmg||S|D|||
|scal||S|D|C/Cs|Z/Zd|
|swap||S|D|C|Z|
|gbmv||||||
|gemv||||||
|ger||||||
|sbmv||||||
|spmv||||||
|spr||||||
|spr2||||||
|symv||||||
|syr||||||
|syr2||||||
|tbmv||||||
|tbsv||||||
|tpmv||||||
|tpsv||||||
|trmv||||||
|trsv||||||
|hemv||||||
|hbmv||||||
|hpmv||||||
|her||||||
|her2||||||
|hpr||||||
|hpr2||||||
|gemm||||||
|gemm3m||||||
|gemmBatched||||||
|gemmStridedBatched||||||
|symm||||||
|syrk||||||
|syr2k||||||
|syrkx||||||
|trmm||||||
|trsm||||||
|trsmBatched||||||
|hemm||||||
|herk||||||
|her2k||||||
|herkx||||||
|geam||||||
|dgmm||||||
|getrfBatched||||||
|getrsBatched||||||
|getriBatched||||||
|matinvBatched||||||
|geqrfBatched||||||
|gelsBatched||||||
|tpttr||||||
|trttp||||||
|gemmEx||||||
|gemm||||||
|hemm||||||
|symm||||||
|syrk||||||
|syr2k||||||
|syrkx||||||
|herk||||||
|her2k||||||
|herkx||||||
|trsm||||||
|trmm||||||
|spmm||||||
