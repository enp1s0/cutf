# nvrtc - NVIDIA Runtime Compilation

## Sample
```cpp
#include <cutf/memory.hpp>
#include <cutf/error.hpp>
#include <cutf/nvrtc.hpp>

int main(){
	const std::size_t N = 1 << 8;

	auto hAB = cutf::memory::get_host_unique_ptr<float>(N);
	for(auto i = decltype(N)(0); i < N; i++) hAB.get()[i] = static_cast<float>(i);
	auto dA = cutf::memory::get_device_unique_ptr<float>(N);
	auto dB = cutf::memory::get_device_unique_ptr<float>(N);
	cutf::memory::copy(dB.get(), hAB.get(), N);

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

- output  
PTX code `std::string` of `function_code`

- exception  
throw `std::runtime_error` if anything happens.

### cutf::nvrtc::get_function
- input

|  name | type | default | description |
|:------|:-----|:--------|:------------|
|`ptx_code`|`std::string`|| ptx code   |
|`function_name` | `std::string` || function name|
|`device_id` | `unsigned int` |`0`| target device id of compiling |

- output   
`CUfunction`

- exception  
throw `std::runtime_error` if anything happens.

### cutf::nvrtc::launch_function
- input

|  name | type | default | description |
|:------|:-----|:--------|:------------|
|`function`|`CUfunction`||function to launch|
|`argument_pointers`|`std::vector<void*>`||`function` argument pointers|
|`grid`|`dim3`||grid|
|`block`|`dim3`||block|
|`stream`|`CUstream`|`nullptr`|stream id|
|`shared_memory_size`|`unsigned int`|0|dynamic shared memory size per thread block in bytes|

- output  
`void`

- exception  
throw `std::runtime_error` if anything happens.
