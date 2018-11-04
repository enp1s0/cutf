# cutf - CUDA Template Functions
The library of the CUDA/C++ Otaku, by the CUDA/C++ Otaku(?), for the CUDA/C++ Otaku shall not perish from the earth.

## Example
```cpp
#include <cutf/math.cuh>
#include <cutf/type.cuh>

constexpr float PI = 3.f;

template <class T, int N>
__global__ void kernel(T* const output, const T* const input){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= N ) return;
	
	output[tid] = mtk::cuda::math::sin( __ldg(input + tid) * mtk::cuda::type::cast<T>(PI) );
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
