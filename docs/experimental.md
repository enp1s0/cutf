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
