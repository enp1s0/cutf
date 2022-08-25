#include <iostream>
#include <cutf/math.hpp>

template <class T>
void check_fp(
		const T hex
		) {
	const auto v = cutf::experimental::fp::reinterpret_as_fp(hex);
	std::printf("[half] 0x%x (%s)\n",
			hex,
			cutf::math::isinf(v) ? "inf" : (cutf::math::isnan(v) ? "nan" : "normal")
			);
}

int main() {
	check_fp<std::uint16_t>(0xfff1);
	check_fp<std::uint16_t>(0xfc00);
	check_fp<std::uint16_t>(0xfb00);
}
