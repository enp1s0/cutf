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
	auto h_out = mtk::cuda::memory::get_device_unique_ptr<T>(N);

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
