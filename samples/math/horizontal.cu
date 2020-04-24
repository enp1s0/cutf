#include <iostream>
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

__global__ void kernel_add(half* const res, const half2 a) {
	res[0] = cutf::math::horizontal::add(a);
}

__global__ void kernel_mul(half* const res, const half2 a) {
	res[0] = cutf::math::horizontal::mul(a);
}

__global__ void kernel_min(half* const res, const half2 a) {
	res[0] = cutf::math::horizontal::min(a);
}

__global__ void kernel_max(half* const res, const half2 a) {
	res[0] = cutf::math::horizontal::max(a);
}

void print(const half a, const std::string str) {
	std::printf("%20s : %e\n", str.c_str(), __half2float(a));
}

int main() {
	auto h = cutf::memory::get_host_unique_ptr<half>(1);

	kernel_add<<<1, 1>>>(h.get(), __float22half2_rn({4.0f, 2.0f}));
	cudaDeviceSynchronize();
	print(*h.get(), "add");
	cudaDeviceSynchronize();

	kernel_mul<<<1, 1>>>(h.get(), __float22half2_rn({4.0f, 2.0f}));
	cudaDeviceSynchronize();
	print(*h.get(), "mul");
	cudaDeviceSynchronize();

	kernel_max<<<1, 1>>>(h.get(), __float22half2_rn({4.0f, 2.0f}));
	cudaDeviceSynchronize();
	print(*h.get(), "max");
	cudaDeviceSynchronize();

	kernel_min<<<1, 1>>>(h.get(), __float22half2_rn({4.0f, 2.0f}));
	cudaDeviceSynchronize();
	print(*h.get(), "min");
	cudaDeviceSynchronize();
}
