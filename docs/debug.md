# cutf Debug Functions Reference

## matrix
```cpp
cutf::debug::matrix::*
```
- `print_matrix` : This function prints a matrix.

## matrix
```cpp
cutf::debug::fp::*
```

- `bitstring_t` : This structure contains same size integer type of a given floating point type.

### Example
```cpp
using bitstring_t = typename cutf::debug::fp::bitstring_t<half>::type;
// bitstring_t = uint16_t
```

## print
```cpp
cutf::debug::print::*
```

- `print_bin` : This function prints bitstring of an input variable
- `print_hex` : This function prints hex-code of an input variable

### Supported types
- Integer : `uint8_t`, `uint16_t`, `uint32_t`, `uint64_t`
- Floating point : `half`, `float`, `double`
