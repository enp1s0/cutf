#include <iostream>
#include <cutf/curand_kernel.hpp>
#include <cutf/memory.hpp>

template <class T>
__global__ void kernel(
		T* const ptr
		) {
	curandState_t state;
	curand_init(0, 0, 0, &state);

	ptr[0] = cutf::curand_kernel::log_normal<T>(&state, 0, 1);
	ptr[0] = cutf::curand_kernel::normal<T>    (&state);
	ptr[0] = cutf::curand_kernel::uniform<T>   (&state);
}


template <class T>
void test() {
	auto mem_uptr = cutf::memory::get_host_unique_ptr<T>(1);
	kernel<<<1, 1>>>(mem_uptr.get());
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
}

int main() {
	test<float >();
	test<double>();
}
