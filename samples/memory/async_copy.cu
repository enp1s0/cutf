#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/stream.hpp>

constexpr std::size_t N = 1 << 17;
constexpr std::size_t threads_per_block = 1 << 8;

template <class T, std::size_t N>
__global__ void kernel(T* const C, const T* const A, const T* const B){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;

	C[tid] = A[tid] + B[tid];
}

using compute_t = float;
int main(){
	auto dA = cutf::cuda::memory::get_device_unique_ptr<compute_t>(N);
	auto dB = cutf::cuda::memory::get_device_unique_ptr<compute_t>(N);
	auto dC = cutf::cuda::memory::get_device_unique_ptr<compute_t>(N);
	auto hA = cutf::cuda::memory::get_host_unique_ptr<compute_t>(N);
	auto hB = cutf::cuda::memory::get_host_unique_ptr<compute_t>(N);
	auto hC = cutf::cuda::memory::get_host_unique_ptr<compute_t>(N);

	auto streamA = cutf::cuda::stream::get_stream_unique_ptr();
	auto streamB = cutf::cuda::stream::get_stream_unique_ptr();


	for(auto i = decltype(N)(0); i < N; i++){
		hA.get()[i] = cutf::cuda::type::cast<compute_t>(1.0f * i / N);
		hB.get()[i] = cutf::cuda::type::cast<compute_t>(1.0f * (N - i) / N);
	}

	cutf::cuda::memory::copy_async(dA.get(), hA.get(), N, *streamA.get());
	cutf::cuda::memory::copy_async(dB.get(), hB.get(), N, *streamB.get());

	cudaStreamSynchronize(*streamA.get());
	cudaStreamSynchronize(*streamB.get());

	kernel<compute_t, N><<<(N + threads_per_block - 1) / threads_per_block, threads_per_block>>>(dC.get(), dA.get(), dB.get());

	cutf::cuda::memory::copy(hC.get(), dC.get(), N);
}
