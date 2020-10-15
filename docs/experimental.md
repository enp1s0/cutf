# cutf Experimental Functions Reference

## tf32
```cpp
cutf::experimental::tf32::*
```

- `tf32_t` : TF32 type
- `to_tf32` : Converting `float` to `tf32_t`

### Casting to TF32 using cutf::type::cast
`cutf::type::cast<nvcuda::wmma::precision::tf32>(x)` converts FP16/32/64 to TF32 using this function if CC < 8.
Otherwise it uses a PTX instruction `cvt.rna.tf32.f32`.

## cut_mantissa
```cpp
cutf::experimental::mantissa::cut_mantissa<mantissa_length, rounging = cutf::rounding::rr>(v : float)
cutf::experimental::mantissa::cut_mantissa<mantissa_length, rounging = cutf::rounding::rr>(v : double)
```

This function cuts mantissa of FP32/FP64 value `v`.  
`to_tf32` is an alias of `cut_mantissa<10, cutf::rounding::rr>(:float)`.

### Supported rounding
- `cutf::rounding::rz`
- `cutf::rounding::rr`
- `cutf::rounding::rb`

## min_exponent
```cpp
// T = half / float / double
cutf::experimental::exponent::min_exponent<T>(T v, int min_exponent)
```

This function returns zero if the exponent of `v` id smaller than `min_exponent` else `v`.

## mask_XXXX
```cpp
cutf::experimental::fp::mask_mantissa(T fp)
cutf::experimental::fp::mask_exponent(T fp)
cutf::experimental::fp::mask_sign(T fp)
```

These functions get each part of floating point values like below.
```
Float (-)
[original] 10111110100001110101001110011101
[sign    ] 10000000000000000000000000000000
[exponent] 00111110100000000000000000000000
[mantissa] 00000000000001110101001110011101
```
