#include <iostream>
#include <cutf/math.hpp>

template <class T>
__device__ __host__ void check_fp(
		const T hex,
		const char* const residence = "host"
		) {
	using fp_t = typename cutf::experimental::fp::same_size_fp<T>::type;
	const auto v = cutf::experimental::fp::reinterpret_as_fp(hex);
	std::printf("[%s: fp%lu] 0x%x (%s)\n",
			residence,
			sizeof(T) * 8,
			hex,
			cutf::math::isnan<fp_t>(v) ? "nan" : (cutf::math::isinf<fp_t>(v) ? "inf" : "normal")
			);
}

template <class T>
__global__ void check_fp_kernel(
		const T hex
		) {
	check_fp(hex, "device");
}

int main() {
	check_fp<std::uint16_t>(0xfff1);
	check_fp<std::uint16_t>(0xfc00);
	check_fp<std::uint16_t>(0xfb00);
	check_fp<std::uint32_t>(0xfff10000);
	check_fp<std::uint32_t>(0xff800000);
	check_fp<std::uint32_t>(0xff700000);

	check_fp_kernel<std::uint16_t><<<1, 1>>>(0xfff1);
	check_fp_kernel<std::uint16_t><<<1, 1>>>(0xfc00);
	check_fp_kernel<std::uint16_t><<<1, 1>>>(0xfb00);
	check_fp_kernel<std::uint32_t><<<1, 1>>>(0xfff10000);
	check_fp_kernel<std::uint32_t><<<1, 1>>>(0xff800000);
	check_fp_kernel<std::uint32_t><<<1, 1>>>(0xff700000);

	cudaDeviceSynchronize();
}
