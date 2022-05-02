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
	const auto input = __float22half2_rn({4.0f, 2.0f});
	auto h = cutf::memory::get_host_unique_ptr<half>(1);

	std::printf("----\n");
	kernel_add<<<1, 1>>>(h.get(), input);
	cudaDeviceSynchronize();
	print(*h.get(), "add (d)");
	print(cutf::math::horizontal::add(input), "add (h)");
	cudaDeviceSynchronize();

	std::printf("----\n");
	kernel_mul<<<1, 1>>>(h.get(), input);
	cudaDeviceSynchronize();
	print(*h.get(), "mul (d)");
	print(cutf::math::horizontal::mul(input), "mul (h)");
	cudaDeviceSynchronize();

	std::printf("----\n");
	kernel_max<<<1, 1>>>(h.get(), input);
	cudaDeviceSynchronize();
	print(*h.get(), "max (d)");
	print(cutf::math::horizontal::max(input), "max (h)");
	cudaDeviceSynchronize();

	std::printf("----\n");
	kernel_min<<<1, 1>>>(h.get(), input);
	cudaDeviceSynchronize();
	print(*h.get(), "min (d)");
	print(cutf::math::horizontal::min(input), "min (h)");
	cudaDeviceSynchronize();
}
