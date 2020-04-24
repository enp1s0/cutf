# CUDA Functions
## math
CUDA built-in unary math functions. (SFU)
```cpp
// x : half, half2, float, double
cutf::math::[operation](x);
```
|Operation| description |
|:--------|:------------|
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

### cutf original function
|Operation| description |
|:--------|:------------|
|sign|`if` $`x > 0`$ `then` $`1`$ `else` $`-1`$|

### horizontal operators of `half2`
|Operation|
|:--------|
|add      |
|mul      |
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
|cast|`half`,`float`,`double` casts each other|
|reinterpret|reinterpret cast|
|rcast|rounding cast|

### rounding
| Rounding type | description |
|:--------------|:------------|
|`cutf::type::rounding::rd`|round-down mode|
|`cutf::type::rounding::rn`|round-to-nearest-even mode|
|`cutf::type::rounding::ru`|round-up mode|
|`cutf::type::rounding::rz`|round-towards-zero mode|

## memory
```cpp
auto dA = cutf::memory::get_device_unique_ptr<type>(N);
auto hA = cutf::memory::get_host_unique_ptr<type>(N);
cutf::memory::copy(dst_ptr, src_ptr, N);
```

| Function | description |
|:--------------|:------------|
|`cutf::memory::get_device_unique_ptr`|`cudaMalloc` and returns `std::unique_ptr`|
|`cutf::memory::get_host_unique_ptr`|`cudaMallocHost` and returns `std::unique_ptr`|
|`cutf::memory::copy`|`cudaMemcpy` with `cudaMemcpyDefault`|

All functions whould throw runtime exception if anything should happen.
