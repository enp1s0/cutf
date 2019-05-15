#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

using compute_t = float;
const std::size_t N = 1 << 10;
const std::size_t threads_per_block = 1 << 6;

namespace{
template <class T, std::size_t N>
__global__ void cos_kernel(T* const dst, const T* const src){
	const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid >= N) return;

	dst[tid] = cutf::math::cos(src[tid]);
}
}

int main(){
	auto host_src = cutf::memory::get_host_unique_ptr<compute_t>(N);
	auto host_dst = cutf::memory::get_host_unique_ptr<compute_t>(N);
	auto device_src = cutf::memory::get_device_unique_ptr<compute_t>(N);
	auto device_dst = cutf::memory::get_device_unique_ptr<compute_t>(N);

	for(auto i = decltype(N)(0); i < N; i++){
		host_src.get()[i] = cutf::type::cast<compute_t>(static_cast<float>(i) / N);
	}

	cutf::memory::copy(device_src.get(), host_src.get(), N);

	cos_kernel<compute_t, N><<<(N + threads_per_block - 1)/threads_per_block, threads_per_block>>>(device_dst.get(), device_src.get());

	cutf::memory::copy(host_dst.get(), device_dst.get(), N);
}
