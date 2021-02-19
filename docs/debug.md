# cutf Debug Functions Reference

## fp namespace
```cpp
cutf::debug::fp::*
```

- `bitstring_t` : This structure contains same size integer type of a given floating point type.

### Example
```cpp
using bitstring_t = typename cutf::debug::fp::bitstring_t<half>::type;
// bitstring_t = uint16_t
```

## print namespace
```cpp
cutf::debug::print::*
```

### floating point (debug/fp.hpp)
- `print_bin` : This function prints bitstring of an input variable
- `print_hex` : This function prints hex-code of an input variable

Supported types are below.

- Integer : `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
- Floating point : `half`, `float`, `double`

### matrix (debug/matrix.hpp)
- `print_matrix` : This function prints a matrix.
- `print_numpy_matrix` : This function prints a matrix as a numpy matrix format

#### Example
```cpp
cutf::debug::print::print_matrix(mat_ptr, M, N, ldm, "mat_a");
cutf::debug::print::print_numpy_matrix(mat_ptr, M, N, ldm, "mat_a");
```

### fragment (debug/fragment.hpp)
- `print_fragment` : This function prints each element of a fragment.

#### Example
```cpp
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, N, N, N, half, nvcuda::wmma::col_major> fragment;
nvcuda::wmma::load_matrix_sync(fragment, mat, N);
cutf::debug::print::print_fragment(fragment, "frag");
```

## clock_breakdown

See [clock_breakdown](clock_breakdown.md).
