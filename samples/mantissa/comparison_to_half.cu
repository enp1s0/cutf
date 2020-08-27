#include <iostream>
#include <random>
#include <cutf/type.hpp>
#include <cutf/experimental/mantissa.hpp>

int main() {
	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	const auto fp32 = dist(mt);
	const auto fp16 = cutf::type::cast<float>(cutf::type::cast<half>(fp32));
	const auto tf32_rz = cutf::experimental::mantissa::cut_mantissa<10, cutf::rounding::rz>(fp32);
	const auto tf32_rn = cutf::experimental::mantissa::cut_mantissa<10, cutf::rounding::rn>(fp32);
	const auto tf32_rr = cutf::experimental::mantissa::cut_mantissa<10, cutf::rounding::rr>(fp32);
	const auto tf32_rb = cutf::experimental::mantissa::cut_mantissa<10, cutf::rounding::rb>(fp32);

	std::printf("fp32    %08x\n", cutf::type::reinterpret<unsigned>(fp32));
	std::printf("fp16    %08x\n", cutf::type::reinterpret<unsigned>(fp16));
	std::printf("tf32_rz %08x\n", cutf::type::reinterpret<unsigned>(tf32_rz));
	std::printf("tf32_rn %08x\n", cutf::type::reinterpret<unsigned>(tf32_rn));
	std::printf("tf32_rr %08x\n", cutf::type::reinterpret<unsigned>(tf32_rr));
	std::printf("tf32_rb %08x\n", cutf::type::reinterpret<unsigned>(tf32_rb));
}
