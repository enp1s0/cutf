#include <iostream>
#include <cutf/arithmetic.hpp>
#include <cutf/debug/type.hpp>

template <class T>
void test_real() {
	std::printf("# %s (%s)\n", __func__, cutf::debug::type::get_type_name<T>());
	const T a = 3.14 / 3, b = 2.76 / 5, c = 1.65 / 7;

	const auto add_v = cutf::type::cast<double>(cutf::arithmetic::add(a, b));
	const double add_ref = 1.59866666667;
	const auto add_error = std::abs((add_v - add_ref) / add_ref);

	const auto sub_v = cutf::type::cast<double>(cutf::arithmetic::sub(a, b));
	const double sub_ref = 0.49466666666;
	const auto sub_error = std::abs((sub_v - sub_ref) / sub_ref);

	const auto mul_v = cutf::type::cast<double>(cutf::arithmetic::mul(a, b));
	const double mul_ref = 0.57776;
	const auto mul_error = std::abs((mul_v - mul_ref) / mul_ref);

	const auto mad_v = cutf::type::cast<double>(cutf::arithmetic::mad(a, b, c));
	const double mad_ref = 0.81347428571;
	const auto mad_error = std::abs((mad_v - mad_ref) / mad_ref);

	std::printf("TEST(ADD) : %s\n", (add_error < 1e-5) ? "OK" : "NG");
	std::printf("TEST(SUB) : %s\n", (sub_error < 1e-5) ? "OK" : "NG");
	std::printf("TEST(MUL) : %s\n", (mul_error < 1e-5) ? "OK" : "NG");
	std::printf("TEST(MAD) : %s\n", (mad_error < 1e-5) ? "OK" : "NG");
}

template <class T>
void test_complex() {
	std::printf("# %s (%s)\n", __func__, cutf::debug::type::get_type_name<T>());
	const T a = cutf::type::make_complex<T>(1. / 3, 4. / 9);
	const T b = cutf::type::make_complex<T>(1. / 5, 4. / 7);
	const T c = cutf::type::make_complex<T>(1. / 7, 4. / 3);

	const auto add_v = cutf::type::cast<cuDoubleComplex>(cutf::arithmetic::add(a, b));
	const auto add_ref = cutf::type::make_complex<cuDoubleComplex>(0.53333333333, 1.01587301587);
	const auto add_error = std::sqrt(cutf::arithmetic::abs2(cutf::arithmetic::sub(add_v, add_ref)) / cutf::arithmetic::abs2(add_ref));

	const auto sub_v = cutf::type::cast<cuDoubleComplex>(cutf::arithmetic::sub(a, b));
	const auto sub_ref = cutf::type::make_complex<cuDoubleComplex>(0.13333333333, -0.12698412698);
	const auto sub_error = std::sqrt(cutf::arithmetic::abs2(cutf::arithmetic::sub(sub_v, sub_ref)) / cutf::arithmetic::abs2(sub_ref));

	const auto mul_v = cutf::type::cast<cuDoubleComplex>(cutf::arithmetic::mul(a, b));
	const auto mul_ref = cutf::type::make_complex<cuDoubleComplex>(-0.1873015873, 0.27936507936);
	const auto mul_error = std::sqrt(cutf::arithmetic::abs2(cutf::arithmetic::sub(mul_v, mul_ref)) / cutf::arithmetic::abs2(mul_ref));

	const auto mad_v = cutf::type::cast<cuDoubleComplex>(cutf::arithmetic::mad(a, b, c));
	const auto mad_ref = cutf::type::make_complex<cuDoubleComplex>(-0.04444444444, 1.6126984127);
	const auto mad_error = std::sqrt(cutf::arithmetic::abs2(cutf::arithmetic::sub(mad_v, mad_ref)) / cutf::arithmetic::abs2(mad_ref));

	std::printf("TEST(ADD) : %s\n", (add_error < 1e-5) ? "OK" : "NG");
	std::printf("TEST(SUB) : %s\n", (sub_error < 1e-5) ? "OK" : "NG");
	std::printf("TEST(MUL) : %s\n", (mul_error < 1e-5) ? "OK" : "NG");
	std::printf("TEST(MAD) : %s\n", (mad_error < 1e-5) ? "OK" : "NG");
}

int main() {
	test_real<double>();
	test_real<float >();

	test_complex<cuDoubleComplex>();
	test_complex<cuComplex      >();
}
