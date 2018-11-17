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
