#include <iostream>
#include <cutf/debug/fp.hpp>

int main(){
	cutf::debug::print::print_bin(1.0f / 3.0f, true);
	cutf::debug::print::print_bin(1.0 / 3.0, true);
	cutf::debug::print::print_hex(1.0f / 3.0f, true);
	cutf::debug::print::print_hex(1.0 / 3.0, true);

	std::printf("# int128 test\n");
	cutf::debug::print::print_hex<__int128_t> (static_cast<__uint128_t>(~0lu) << 32, true);
	cutf::debug::print::print_hex<__uint128_t>(static_cast<__uint128_t>(~0lu) << 32, true);
	cutf::debug::print::print_bin<__int128_t> (static_cast<__uint128_t>(~0lu) << 32, true);
	cutf::debug::print::print_bin<__uint128_t>(static_cast<__uint128_t>(~0lu) << 32, true);
}
