# CUDA Functions
## math
```cpp
cutf::cuda::math::[operation](x);
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

## type
```cpp
cutf::cuda::type::cast<type>(x);
cutf::cuda::type::reinterpret<type>(x);
cutf::cuda::type::rcast<type, rounding>(x);
```

### description

|Cast| description |
|:--------|:------------|
|cast|`half`,`float`,`double` casts each other|
|reinterpret|reinterpret cast|
|rcast|rounding cast|

### rounding
| Rounding type | description |
|:--------------|:------------|
|`cutf::cuda::type::rounding::rd`|round-down mode|
|`cutf::cuda::type::rounding::rn`|round-to-nearest-even mode|
|`cutf::cuda::type::rounding::ru`|round-up mode|
|`cutf::cuda::type::rounding::rz`|round-towards-zero mode|

## memory
```cpp
auto dA = cutf::cuda::memory::get_device_unique_ptr<type>(size);
auto hA = cutf::cuda::memory::get_host_unique_ptr<type>(size);
cutf::cuda::memory::copy(dst_ptr, src_ptr, size);
```
