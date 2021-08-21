#include <cutf/type.hpp>
#include <cutf/experimental/fp.hpp>
#include <cutf/debug/print.hpp>
#include <iostream>
#include <random>

template <class T, class MASK_T = typename cutf::experimental::fp::same_size_uint<T>::type>
void test_mask(const T fp, const MASK_T mask) {
	const auto org = cutf::experimental::fp::reinterpret_as_uint(fp);
	const auto masked = static_cast<MASK_T>(org & mask);
	const auto masked_fp = cutf::experimental::fp::reinterpret_as_fp<MASK_T>(masked);

	std::printf("[raw      ] ");cutf::debug::print::print_bin(org);
	std::printf("[masked   ] ");cutf::debug::print::print_bin(masked);
	std::printf("[masked_fp] ");cutf::debug::print::print_bin(masked_fp);
}

int main() {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dist(0.0f, 1.0f);
	const auto fp = dist(mt);

	test_mask(
			cutf::type::cast<double>(fp),
			static_cast<typename cutf::experimental::fp::same_size_uint<double>::type>(0xffff)
			);

	test_mask(
			cutf::type::cast<float >(fp),
			static_cast<typename cutf::experimental::fp::same_size_uint<float >::type>(0xffff)
			);

	test_mask(
			cutf::type::cast<half  >(fp),
			static_cast<typename cutf::experimental::fp::same_size_uint<half  >::type>(0xffff)
			);
}
