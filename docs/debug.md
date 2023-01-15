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
- `print_matrix_from_host` : This function prints a matrix on the device memory from host code.
- `print_matrix_debug` : This function prints a matrix in hex.
- `print_matrix_hex_from_host` : This function prints a matrix on the device memory from host code in hex.
- `print_numpy_matrix` : This function prints a matrix as a numpy matrix format.
- `print_numpy_matrix_from_host` : This function prints a matrix as a numpy matrix format on the device memory from host code.

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

## time_breakdown
### Sample code
```cpp
cutf::debug::time_breakdown::profiler profiler(cuda_stream);

// put some operations between start and stop functions
profiler.start_timer_sync("malloc_a");
cudaMalloc(...);
profiler.stop_timer_sync("malloc_a");

// give some operations to measure function as a lambda function
profiler.measure("malloc_b", [&](){cudaMalloc(...);});

// output time breakdown
profiler.print_result();
```

### Sample result
```
# cutf time breakdown result (Total:      3.396 [ms])
        Name    Total [ms]                    N   Avg [ms]   Min [ms]   Max [ms]
  cudaMemcpy         2.225 ( 65.52%)          1      2.225      2.225      2.225
add_1_kernel         1.070 ( 31.52%)        100      0.011      0.010      0.015
  cudaMalloc         0.081 (  2.39%)          1      0.081      0.081      0.081
 init_kernel         0.019 (  0.57%)          1      0.019      0.019      0.019
```

## clock_breakdown

See [clock_breakdown](clock_breakdown.md).
