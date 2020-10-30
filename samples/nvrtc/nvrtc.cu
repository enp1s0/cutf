#include <iostream>
#include <random>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cuda.hpp>
#include <cutf/nvrtc.hpp>

int main(){
	const std::size_t N = 1 << 8;
	const std::string code = R"(
extern "C"
__global__ void kernel(float *a, float *b){
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	a[tid] = b[tid];
}
)";
	CUTF_CHECK_ERROR(cuInit(0));
	auto cu_context = cutf::cu::get_context_unique_ptr();
	cutf::cu::create_context(cu_context.get(), 0);
	auto cu_module = cutf::cu::get_module_unique_ptr();

	const auto ptx_code = cutf::nvrtc::get_ptx(
				"kernel.cu",
				code,
				{"--arch=sm_60"},
				{},
				false
			);

	const auto function = cutf::nvrtc::get_function(
			ptx_code,
			"kernel",
			cu_module.get()
			);

	std::cout<<"/* -- PTX" <<std::endl
			<<ptx_code<<std::endl
			<<" -- */"<<std::endl;

	auto hAB = cutf::memory::get_host_unique_ptr<float>(N);
	for(auto i = decltype(N)(0); i < N; i++) hAB.get()[i] = static_cast<float>(i);
	auto dA = cutf::memory::get_device_unique_ptr<float>(N);
	auto dB = cutf::memory::get_device_unique_ptr<float>(N);
	cutf::memory::copy(dB.get(), hAB.get(), N);

	const float * dA_ptr = dA.get();
	const float * dB_ptr = dB.get();

	std::cout<<"# -- kernel launch" <<std::endl;

	cutf::nvrtc::launch_function(
			function,
			{&dA_ptr, &dB_ptr},
			N,
			1
			);

	cutf::memory::copy(hAB.get(), dA.get(), N);

	std::cout<<"/* -- kernel result" <<std::endl;
	for(auto i = decltype(N)(0); i < N; i++) {
		std::cout<<hAB.get()[i] << " ";
		if(i % 8 == 7)
			std::cout<<std::endl;
	}
	std::cout<<" -- */"<<std::endl;
}
