# nvrtc - NVIDIA Runtime Compilation

## Sample
```cpp
#include <cutf/memory.hpp>
#include <cutf/error.hpp>
#include <cutf/nvrtc.hpp>

int main(){
	const std::size_t N = 1 << 8;

	auto hAB = cutf::cuda::memory::get_host_unique_ptr<float>(N);
	for(auto i = decltype(N)(0); i < N; i++) hAB.get()[i] = static_cast<float>(i);
	auto dA = cutf::cuda::memory::get_device_unique_ptr<float>(N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<float>(N);
	cutf::cuda::memory::copy(dB.get(), hAB.get(), N);

	const float * dA_ptr = dA.get();
	const float * dB_ptr = dB.get();

	// NVRTC
	const std::string code = R"(
extern "C"
__global__ void kernel(float *a, float *b){
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	a[tid] = b[tid];
}
)";
	const auto ptx_code = cutf::nvrtc::get_ptx(
				"kernel.cu",
				code,
				{"--arch=sm_60"},
				{},
				false
			);

	const auto function = cutf::nvrtc::get_function(
			ptx_code,
			"kernel"
			);

	cutf::nvrtc::launch_function(
			function,
			{&dA_ptr, &dB_ptr},
			N,
			1
			);
}

```

## Functions
###  cutf::nvrtc::get_ptx
- input

|  name | type | default | description |
|:------|:-----|:--------|:------------|
|`source_name`|`std::string`|| source code name you like (e.g. kernel.cu) |
|`function_code` | `std::string` || kernel source code |
|`compile_options` | `std::vector<std::string>` |`{}`| compile option list (e.g. `{"--arch=sm70", ...}`)|
|`headers` | `std::vector<std::pair<std::string, std::string>>` |`{}`| header list. pair of (header name, header sources)|
|`print_compile_log` | `bool` |`false`| if `true`, print compiling log to `stdout`|

- output \\
PTX code `std::string` of `function_code`

- exception \\
throw `std::runtime_error` if anything happens.
