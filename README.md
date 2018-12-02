<img src="./docs/cutf-logo.png" width="150">


# cutf - CUDA Template Functions
*The library of the CUDA/C++ Otaku, by the CUDA/C++ Otaku(?), for the CUDA/C++ Otaku shall not perish from the earth.*

## Introduction
cutf is a tiny CUDA template library.

- header file only
- at least C++11

## Development
- release/devel : [GitLab momo86.net - mutsuki/cutf](https://gitlab.momo86.net/mutsuki/cutf)
- release : [GitHub - gonmoec/cutf](https://github.com/gonmoec/cutf)

## Example
```cpp
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
constexpr float PI = 3.f;
constexpr std::size_t N = 15;

template <class T, int N>
__global__ void kernel_example(T* const output, const T* const input){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= N ) return;

	output[tid] = cutf::cuda::math::sin( __ldg(input + tid) * cutf::cuda::type::cast<T>(PI) );
}

using T = float;
int main(){
	auto in = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto out = cutf::cuda::memory::get_device_unique_ptr<T>(N);
	auto h_out = cutf::cuda::memory::get_host_unique_ptr<T>(N);

	kernel_example<T, N><<<(N+15)/16,16>>>(out.get(), in.get());

	cutf::cuda::memory::copy(h_out.get(), out.get(), N);
}
```

## Namespace structure
```
cutf 
├─ cublas
│  └─ error
├─ cuda
│  ├─ device
│  ├─ error
│  ├─ math
│  ├─ memory
│  └─ type
│     └─ rounding
├─ driver
│  └─ error
├─ cublas
│  └─ error
└─ nvrtc
   └─ error
```

## Smart pointers
[Smart pointers Reference](./docs/smart_ptr.md)

## CUDA Functions
[cutf CUDA Functions Reference](./docs/cuda.md)

## cuBLAS Functions
[cutf cuBLAS Functions Reference](./docs/cublas.md)

## NVRTC Functions
[cutf NVRTC Functions Reference](./docs/nvrtc.md)

## License
Copyright (c) 2018 mutsuki (gonmoec)  
Released under the MIT license  
<img src="http://momo86.net/ipsolab.svg" width="120">
