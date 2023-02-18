<img src="./docs/cutf-logo.png" width="150">


# cutf - CUDA Template Functions
*The library of the CUDA/C++ Otaku, by the CUDA/C++ Otaku(?), for the CUDA/C++ Otaku shall not perish from the earth.*

**Warning!**

This library is under developing.
Destructive changes may occur.

## Introduction
cutf is a tiny CUDA template library.

- header file only
- at least C++14

## Development
- release/devel : [GitLab momo86.net - mutsuki/cutf](https://gitlab.momo86.net/mutsuki/cutf)
- release : [GitHub - enp1s0/cutf](https://github.com/enp1s0/cutf)

## Example
```cpp
// sample.cu
// Compile:
// nvcc -I/path/to/cutf/include/ sample.cu ...
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/error.hpp>
#include <cutf/memory.hpp>
constexpr float PI = 3.f;
constexpr std::size_t N = 15;

template <class T, int N>
__global__ void kernel_example(T* const output, const T* const input){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if( tid >= N ) return;

	output[tid] = cutf::math::sin( __ldg(input + tid) * cutf::type::cast<T>(PI) );
}

using T = float;
int main(){
	auto in = cutf::memory::get_device_unique_ptr<T>(N);
	auto out = cutf::memory::get_device_unique_ptr<T>(N);
	auto h_out = cutf::memory::get_host_unique_ptr<T>(N);

	kernel_example<T, N><<<(N+15)/16,16>>>(out.get(), in.get());

	CUTF_CHECK_ERROR(cutf::memory::copy(h_out.get(), out.get(), N));
}
```

## Namespace structure
```
cutf 
├─ cp_async
├─ cublas
├─ cuda
├─ cufft
├─ cupti
├─ curand
├─ curand_kernel
├─ cusolver
├─ cutensor
├─ debug
│  ├─ fp
│  └─ print
├─ device
├─ driver
├─ error
├─ event
├─ experimental
│  └─ fp
├─ graph
├─ math
├─ memory
├─ nvrtc
├─ type
│  └─ rounding
└─ thread
```

## Smart pointers
[Smart pointers Reference](./docs/smart_ptr.md)

## CUDA Functions
[cutf CUDA Functions Reference](./docs/cuda.md)

## cuBLAS Functions
[cutf cuBLAS Functions Reference](./docs/cublas.md)

## cuSOLVER Functions
[cutf cuSOLVER Functions Reference](./docs/cusolver.md)

## NVRTC Functions
[cutf NVRTC Functions Reference](./docs/nvrtc.md)

## Debug functions
[cutf Debug Functions Reference](./docs/debug.md)

## Experimental  functions
[cutf Experimental Functions Reference](./docs/experimental.md)

## License
Copyright (c) 2018 - 2021 tsuki (enp1s0)
Released under the MIT license  
