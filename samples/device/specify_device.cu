#include <iostream>
#include <cutf/device.hpp>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>

constexpr std::size_t N = 1 << 16;
constexpr std::size_t threads_per_block  = 1 << 8;
using compute_t = float;

namespace{
template <class T, std::size_t N>
__global__ void kernel(T* const t){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	const auto v = __ldg(t + tid);
	t[tid] = v * v;
}
}

int main(){
	const auto device_properties = cutf::device::get_properties_vector();
	std::int_fast32_t device_id = 0;
	for(const auto & dp : device_properties){
		std::cout<<"# "<<device_id<<std::endl
			<<"Name          : "<<dp.name<<std::endl
			<<"Global Memory : "<<(dp.totalGlobalMem/(1<<20))<<" MB"<<std::endl;

		CUTF_CHECK_ERROR(cutf::device::use_device(
				device_id,
				[](){
					auto dMem = cutf::memory::get_device_unique_ptr<compute_t>(N);
					auto hMem = cutf::memory::get_host_unique_ptr<compute_t>(N);
					for(auto i = decltype(N)(0); i < N; i++){
						hMem.get()[i] = cutf::type::cast<compute_t>(1.0f * i / N);
					}
					cutf::memory::copy(dMem.get(), hMem.get(), N);
					kernel<compute_t, N><<<(N + threads_per_block - 1)/threads_per_block, threads_per_block>>>(dMem.get());
					cutf::memory::copy(hMem.get(), dMem.get(), N);
				}));
		device_id++;
	}
}
