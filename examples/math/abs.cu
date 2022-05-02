#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/debug/type.hpp>

constexpr std::size_t N = 1 << 4;
constexpr std::size_t threads_per_block = 1 << 7;

namespace{
template <class T>
__global__ void abs_kernel(T* const m, const std::size_t N){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;
	m[tid] = cutf::math::abs(*(m + tid));
}

template <class T>
void test_abs(const std::size_t N){
	std::cout<<"# "<<cutf::debug::type::get_type_name<T>()<<" test --"<<std::endl;
	auto dM = cutf::memory::get_device_unique_ptr<T>(N);
	auto hM = cutf::memory::get_host_unique_ptr<T>(N);
	std::cout<<"m = ";
	for(auto i = decltype(N)(0); i < N; i++){
		hM.get()[i] = cutf::type::cast<T>(N/2.0f - i);
		std::cout<<cutf::type::cast<float>(hM.get()[i])<<" ";
	}
	std::cout<<std::endl;
	cutf::memory::copy(dM.get(), hM.get(), N);

	abs_kernel<T><<<(N + threads_per_block - 1)/threads_per_block, threads_per_block>>>(dM.get(), N);

	cutf::memory::copy(hM.get(), dM.get(), N);

	std::cout<<"|m| = ";
	for(auto i = decltype(N)(0); i < N; i++){
		std::cout<<cutf::type::cast<float>(hM.get()[i])<<" ";
	}
	std::cout<<std::endl;
}

template <>
void test_abs<half2>(const std::size_t N){
	std::cout<<"# "<<cutf::debug::type::get_type_name<half2>()<<" test --"<<std::endl;
	auto dM = cutf::memory::get_device_unique_ptr<half2>(N);
	auto hM = cutf::memory::get_host_unique_ptr<half2>(N);
	std::cout<<"m = ";
	for(auto i = decltype(N)(0); i < N; i++){
		hM.get()[i].x = cutf::type::cast<half>(N/2.0f - i);
		hM.get()[i].y = cutf::type::cast<half>(-(N/2.0f - i));
		std::cout<<cutf::type::cast<float>(hM.get()[i].x)<<" ";
		std::cout<<cutf::type::cast<float>(hM.get()[i].y)<<" ";
	}
	std::cout<<std::endl;
	cutf::memory::copy(dM.get(), hM.get(), N);

	abs_kernel<half2><<<(N + threads_per_block - 1)/threads_per_block, threads_per_block>>>(dM.get(), N);

	cutf::memory::copy(hM.get(), dM.get(), N);

	std::cout<<"|m| = ";
	for(auto i = decltype(N)(0); i < N; i++){
		std::cout<<cutf::type::cast<float>(hM.get()[i].x)<<" ";
		std::cout<<cutf::type::cast<float>(hM.get()[i].y)<<" ";
	}
	std::cout<<std::endl;
}
}

int main(){
	test_abs<half>(N);
	test_abs<half2>(N);
	test_abs<float>(N);
	test_abs<double>(N);
}
