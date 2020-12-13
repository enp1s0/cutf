#include <iostream>
#include <cutf/debug/fp.hpp>

int main(){
	cutf::debug::print::print_bin(1.0f / 3.0f, true);
	cutf::debug::print::print_bin(1.0 / 3.0, true);
}
