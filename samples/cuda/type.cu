#include <iostream>
#include <cutf/type.hpp>

template <class T>
__global__ void print_typename_test() {
	const auto type_string = cutf::type::get_type_name<T>();
	printf("%s\n", type_string);
}

int main() {
	print_typename_test<double><<<1, 1>>>();
	print_typename_test<float ><<<1, 1>>>();
	print_typename_test<half  ><<<1, 1>>>();
	print_typename_test<half2 ><<<1, 1>>>();
	cudaDeviceSynchronize();
}
