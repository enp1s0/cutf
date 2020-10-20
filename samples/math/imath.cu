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
__global__ void maxmin_kernel(T* const max_array, T* const min_array, const T* const m0_array, const T* const m1_array, const std::size_t N){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;

	const auto a = *(m0_array + tid), b = *(m1_array + tid);
	max_array[tid] = cutf::math::max(a, b);
	min_array[tid] = cutf::math::min(a, b);
}

template <class T>
void test_abs(const std::size_t N){
	std::cout<<"# "<<cutf::debug::type::get_type_name<T>()<<" test --"<<std::endl;
	auto dM = cutf::memory::get_device_unique_ptr<T>(N);
	auto hM = cutf::memory::get_host_unique_ptr<T>(N);
	std::cout<<"m = ";
	for(auto i = decltype(N)(0); i < N; i++){
		hM.get()[i] = N / 2 - i;
		std::cout<<hM.get()[i]<<" ";
	}
	std::cout<<std::endl;
	cutf::memory::copy(dM.get(), hM.get(), N);

	abs_kernel<T><<<(N + threads_per_block - 1)/threads_per_block, threads_per_block>>>(dM.get(), N);

	cutf::memory::copy(hM.get(), dM.get(), N);

	std::cout<<"|m| = ";
	for(auto i = decltype(N)(0); i < N; i++){
		std::cout<<hM.get()[i]<<" ";
	}
	std::cout<<std::endl;
}

template <class T>
void test_maxmin(const std::size_t N){
	std::cout<<"# "<<cutf::debug::type::get_type_name<T>()<<" test --"<<std::endl;
	auto dM0 = cutf::memory::get_device_unique_ptr<T>(N);
	auto dM1 = cutf::memory::get_device_unique_ptr<T>(N);
	auto hM0 = cutf::memory::get_host_unique_ptr<T>(N);
	auto hM1 = cutf::memory::get_host_unique_ptr<T>(N);
	std::cout<<"m = ";
	for(auto i = decltype(N)(0); i < N; i++){
		hM0.get()[i] = N / 2 - i;
		hM1.get()[i] = i - N / 2;
		std::cout<<"("<<hM0.get()[i]<<","<<hM1.get()[i]<<") ";
	}
	std::cout<<std::endl;
	cutf::memory::copy(dM0.get(), hM0.get(), N);
	cutf::memory::copy(dM1.get(), hM1.get(), N);

	maxmin_kernel<T><<<(N + threads_per_block - 1)/threads_per_block, threads_per_block>>>(
			dM0.get(), dM1.get(),
			dM0.get(), dM1.get(),
			N
			);

	cutf::memory::copy(hM0.get(), dM0.get(), N);
	cutf::memory::copy(hM1.get(), dM1.get(), N);

	std::cout<<"(max, min) = ";
	for(auto i = decltype(N)(0); i < N; i++){
		std::cout<<"("<<hM0.get()[i]<<","<<hM1.get()[i]<<") ";
	}
	std::cout<<std::endl;
}
}

int main(){
	test_abs<int>(N);
	test_maxmin<int>(N);
}
