#include <iostream>
#include <cutf/memory.hpp>
#include <cutf/type.hpp>
#include <cutf/stream.hpp>

constexpr std::size_t N = 1 << 17;
constexpr std::size_t threads_per_block = 1 << 8;

template <class T, std::size_t N>
__global__ void pow2_kernel(T* const C){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;

	const auto c = __ldg(C + tid);

	C[tid] = c * c;
}

using compute_t = float;
int main(){
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(N);
	auto dB = cutf::memory::get_device_unique_ptr<compute_t>(N);
	auto hA = cutf::memory::get_host_unique_ptr<compute_t>(N);
	auto hB = cutf::memory::get_host_unique_ptr<compute_t>(N);

	auto streamA = cutf::stream::get_stream_unique_ptr();
	auto streamB = cutf::stream::get_stream_unique_ptr();


	for(auto i = decltype(N)(0); i < N; i++){
		hA.get()[i] = cutf::type::cast<compute_t>(1.0f * i / N);
		hB.get()[i] = cutf::type::cast<compute_t>(1.0f * (N - i) / N);
	}

	cutf::memory::copy_async(dA.get(), hA.get(), N, *streamA.get());
	cutf::memory::copy_async(dB.get(), hB.get(), N, *streamB.get());

	pow2_kernel<compute_t, N><<<(N + threads_per_block - 1) / threads_per_block, threads_per_block, 0, *streamA.get()>>>(dA.get());
	pow2_kernel<compute_t, N><<<(N + threads_per_block - 1) / threads_per_block, threads_per_block, 0, *streamB.get()>>>(dB.get());

	cutf::memory::copy_async(dA.get(), hA.get(), N, *streamA.get());
	cutf::memory::copy_async(dB.get(), hB.get(), N, *streamB.get());
}
