#include <iostream>
#include <cutf/debug/fp.hpp>

int main(){
	cutf::debug::fp::print_bin(1.0f / 3.0f, true);
	cutf::debug::fp::print_bin(1.0 / 3.0, true);
}
