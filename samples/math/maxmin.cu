#include <iostream>
#include <cutf/math.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

template <class T>
__global__ void kernel_min(T* const res, const T a, const T b) {
	res[0] = cutf::math::min(a, b);
}

template <class T>
__global__ void kernel_max(T* const res, const T a, const T b) {
	res[0] = cutf::math::max(a, b);
}

void print(const float a, const std::string str) {
	std::printf("%20s : %e\n", str.c_str(), a);
}
void print(const double a, const std::string str) {
	std::printf("%20s : %e\n", str.c_str(), a);
}
void print(const half a, const std::string str) {
	std::printf("%20s : %e\n", str.c_str(), __half2float(a));
}
void print(const half2 a, const std::string str) {
	std::printf("%20s : %e, %e\n", str.c_str(), __half2float(a.x), __half2float(a.y));
}

int main() {
	auto f = cutf::memory::get_host_unique_ptr<float>(1);
	auto d = cutf::memory::get_host_unique_ptr<double>(1);
	auto h = cutf::memory::get_host_unique_ptr<half>(1);
	auto h2 = cutf::memory::get_host_unique_ptr<half2>(1);

	kernel_max<float><<<1, 1>>>(f.get(), 1, 2);
	kernel_max<double><<<1, 1>>>(d.get(), 1, 2);
	kernel_max<half><<<1, 1>>>(h.get(), cutf::type::cast<half>(1), cutf::type::cast<half>(2));
	kernel_max<half2><<<1, 1>>>(h2.get(), __float2half2_rn(1), __float2half2_rn(2));
	cudaDeviceSynchronize();

	print(*f.get(), "max f");
	print(*d.get(), "max d");
	print(*h.get(), "max h");
	print(*h2.get(), "max h2");

	kernel_min<float><<<1, 1>>>(f.get(), 1, 2);
	kernel_min<double><<<1, 1>>>(d.get(), 1, 2);
	kernel_min<half><<<1, 1>>>(h.get(), cutf::type::cast<half>(1), cutf::type::cast<half>(2));
	kernel_min<half2><<<1, 1>>>(h2.get(), __float2half2_rn(1), __float2half2_rn(2));

	print(*f.get(), "min f");
	print(*d.get(), "min d");
	print(*h.get(), "min h");
	print(*h2.get(), "min h2");
}
