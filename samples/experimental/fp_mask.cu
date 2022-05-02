#include <cutf/experimental/fp.hpp>
#include <cutf/debug/fp.hpp>
#include <iostream>
#include <random>

template <class T>
void test_mask(const T fp) {
	const auto e = cutf::experimental::fp::mask_exponent(fp);
	const auto m = cutf::experimental::fp::mask_mantissa(fp);
	const auto s = cutf::experimental::fp::mask_sign(fp);

	std::printf("[raw     ] ");cutf::debug::print::print_bin(fp);
	std::printf("[sign    ] ");cutf::debug::print::print_bin(s);
	std::printf("[exponent] ");cutf::debug::print::print_bin(e);
	std::printf("[mantissa] ");cutf::debug::print::print_bin(m);
}

int main() {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<double> dist(0.0f, 1.0f);
	const auto fp = dist(mt);

	std::printf("Double (+)\n");
	test_mask(fp);
	std::printf("Double (-)\n");
	test_mask(-fp);

	std::printf("Float (+)\n");
	test_mask(static_cast<float>(fp));
	std::printf("Float (-)\n");
	test_mask(static_cast<float>(-fp));
}
