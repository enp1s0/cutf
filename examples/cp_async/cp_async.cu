#include <iostream>
#include <cutf/cp_async.hpp>
#include <cutf/memory.hpp>

template <class T>
__host__ __device__ constexpr unsigned get_size_in_byte();
template <> __host__ __device__ constexpr unsigned get_size_in_byte<float >() {return 4;};
template <> __host__ __device__ constexpr unsigned get_size_in_byte<float2>() {return 8;};
template <> __host__ __device__ constexpr unsigned get_size_in_byte<float4>() {return 16;};

template <class T, unsigned block_size>
__global__ void cp_async_test_kernel(
		T* const dst_ptr,
		const T* const src_ptr
		) {
	__shared__ T smem[block_size];

	cutf::cp_async::cp_async<get_size_in_byte<T>()>(smem + threadIdx.x, src_ptr + threadIdx.x);
	cutf::cp_async::commit();

	cutf::cp_async::wait_all();
	dst_ptr[threadIdx.x] = smem[threadIdx.x];
}

template <class T, unsigned block_size>
void cp_async_test() {
	auto d_input  = cutf::memory::get_device_unique_ptr<T>(block_size);
	auto d_output = cutf::memory::get_device_unique_ptr<T>(block_size);
	auto h_input  = cutf::memory::get_host_unique_ptr<T>(block_size);
	auto h_output = cutf::memory::get_host_unique_ptr<T>(block_size);

	for (unsigned i = 0; i < block_size * get_size_in_byte<T>() / 4; i++) {
		reinterpret_cast<float*>(h_input.get())[i] = i;
	}

	cutf::memory::copy(d_input.get(), h_input.get(), block_size);

	cp_async_test_kernel<T, block_size><<<1, block_size>>>(d_output.get(), d_input.get());

	cutf::memory::copy(h_output.get(), d_output.get(), block_size);

	double max_error = 0;
	for (unsigned i = 0; i < block_size * get_size_in_byte<T>() / 4; i++) {
		const double diff = reinterpret_cast<float*>(h_output.get())[i] - reinterpret_cast<float*>(h_input.get())[i];
		max_error = std::max(std::abs(diff), max_error);
	}

	std::printf("%s[%2u Byte] error = %e\n", __func__, get_size_in_byte<T>(), max_error);
}

int main() {
	cp_async_test<float , 128>();
	cp_async_test<float2, 128>();
	cp_async_test<float4, 128>();
}
