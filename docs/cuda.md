# CUDA Functions
## math
CUDA built-in unary math functions.
```cpp
// x : half, half2, float, double
cutf::math::[operation](x);
```
|Operation| description |
|:--------|:------------|
|abs|$` \|x\| `$|
|ceil|$`\lceil x \rceil`$|
|cos|$`\mathrm{cos} x`$|
|exp|$`\mathrm{e}^{x}`$|
|exp10|$`10^x`$|
|exp2|$`2^x`$|
|floor|$`\lfloor x \rfloor`$|
|log|$`\ln x`$|
|log10|$`\log_{10} x`$|
|log2|$`\log_{2} x`$|
|rcp|$`\frac{1}{x}`$|
|rint|Round input to nearest integer value|
|rsqrt|$`\frac{1}{\sqrt{x}}`$|
|sin|$`\mathrm{sin} x `$|
|sqrt|$`\sqrt{x}`$|
|trunc|	Truncate input argument to the integral part|

### isnan and isinf
```cpp
// x : half, float, double
cutf::math::[operation](x);
```
|Operation| description |
|:--------|:------------|
|isinf| is inf |
|isnan| is nan |

### cutf original function
|Operation| description |
|:--------|:------------|
|sign|`if` $`x > 0`$ `then` $`1`$ `else` $`-1`$|

### horizontal operators for `half2`
|Operation|
|:--------|
|add      |
|mul      |
|max      |
|min      |

### SIMD functions for half2
|Operation|
|:--------|
|max      |
|min      |

Before Ampare architecture there is no `max` and `min` function for `half2`.
These functions are implemented with `__byte_perm` built function.

### math functions for integer
|Operation|
|:--------|
|abs      |
|max      |
|min      |

## type
```cpp
// cast decltype(x) to `type`
cutf::type::cast<type>(x);
cutf::type::reinterpret<type>(x);
cutf::type::rcast<type, rounding>(x);
```

|Cast| description |
|:--------|:------------|
|cast|`half`,`float`,`double`,`tf32` casts each other|
|reinterpret|reinterpret cast|
|rcast|rounding cast|

### type name
```
cutf::type::get_type_name<T>();
```
This function returns the type name (`const char*`).

### rounding
| Rounding type | description |
|:--------------|:------------|
|`cutf::type::rounding::rd`|round-down mode|
|`cutf::type::rounding::rn`|round-to-nearest-even mode|
|`cutf::type::rounding::ru`|round-up mode|
|`cutf::type::rounding::rz`|round-towards-zero mode|

## memory
### Smart pointer
```cpp
auto dA = cutf::memory::get_device_unique_ptr<type>(N);
auto hA = cutf::memory::get_host_unique_ptr<type>(N);
cutf::memory::copy(dst_ptr.get(), src_ptr.get(), N);
```

| Function | description |
|:--------------|:------------|
|`cutf::memory::get_device_unique_ptr`|`cudaMalloc` and returns `std::unique_ptr`|
|`cutf::memory::get_host_unique_ptr`|`cudaMallocHost` and returns `std::unique_ptr`|
|`cutf::memory::copy`|`cudaMemcpy` with `cudaMemcpyDefault`|

All functions would throw runtime exception if anything should happen.

### malloc/free

```cpp
auto dA = cutf::memory::malloc_managed<type>(N);
cutf::memory::free(dA);
```
| mallo | free | remarks |
|:--------------|:------------|:------------|
|`cutf::memory::malloc`| `cutf::memory::free` |  |
|`cutf::memory::malloc_host`| `cutf::memory::free_host` |  |
|`cutf::memory::malloc_managed`| (`cutf::memory::free`) |  |
|`cutf::memory::malloc_async`| `cutf::memory::free_async` | CUDA >= 11.2 |

Note.  
CUDA < 11.2 does not support `cudaMallocAsync` and `cudaFreeAsync`.
By defining `CUTF_DISABLE_MALLOC_ASYNC` before including `memory.hpp`, `malloc_async` and `free_async` use standard `malloc` and `free` with `cudaStreamSynchronize`.

```cuda
#define CUTF_DISABLE_MALLOC_ASYNC
#include <cutf/memory.hpp>

auto ptr = cutf::memory::malloc_async<T>(count, stream);
// It is equivalent to
// CUTF_CHECK_ERROR(cudaStreamSynchronize(stream));
// auto ptr = cutf::memory::malloc<T>(count);
```

## device
```cpp
CUTF_CHECK_ERROR(cutf::device::use_device(
	device_id,
	[]() {
		cudaMalloc(...);
	}));
```

| Function | description |
|:--------------|:------------|
|`cutf::device::get_properties_vector`|Getting the `std::vector` of `cudaDeviceProp`|
|`cutf::device::get_num_devices`|Getting the number of devices|
|`cutf::device::use_device`|Executing lambda function on a specified device|
|`cutf::device::get_device`|Getting device ID|
|`cutf::device::set_device`|Setting device ID|

## thread
```cpp
const auto lane_id = cutf::thread::get_lane_id();
const auto warp_id = cutf::thread::get_warp_id();
constexpr auto warp_size = cutf::thread::warp_size_const;
```

`lane_id` means an unique id for a thread within a warp and `warp_id` means an unique id for a warp within a thread-block.
Thus when you lauch threads with 1D thread block,
- `warp_id` equals to `threadIdx.x / 32`
- `lane_id` equals to `threadIdx.x % 32`

.

This functions get these values from PTX predefines `%warpid` and `%laneid`.

`cutf::thread::warp_size_const` is `constexpr` 32.
CUDA provides `warpSize` to get warp size but it is not const variable and we can't use it for template argument for instance.
