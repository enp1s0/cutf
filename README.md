# cutf - CUDA Template Functions
The library of the CUDA/C++ Otaku, by the CUDA/C++ Otaku(?), for the CUDA/C++ Otaku shall not perish from the earth.

## Example
```cpp
constexpr float PI = 3.f;
template <class T, int N>
__global__ void kernel(T* const output, const T* const input){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= N ) return;
	
	output[tid] = cuda::math::sin( __ldg(input + tid) * cuda::type::cast<T>(PI) );
}
```
