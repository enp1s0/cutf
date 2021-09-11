#include <cutf/cufft.hpp>
#include <cutf/memory.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <random>

using compute_t = float;
using complex_t = std::complex<compute_t>;
using data_t = std::vector<complex_t>;

int main() {
	const unsigned fft_dim[3] = {256, 256, 256};
	const auto num_elements = fft_dim[0] * fft_dim[1] * (fft_dim[2] / 2 + 1);

	std::mt19937 mt(std::random_device{}());
	std::uniform_real_distribution<compute_t> dist(-1, 1);
	data_t host_data(num_elements);
	for (auto& v : host_data) {
		v = {dist(mt), dist(mt)};
	}

	auto device_data = cutf::memory::get_device_unique_ptr<complex_t>(num_elements);
	cutf::memory::copy(device_data.get(), host_data.data(), num_elements);

	auto cufft_handle = cutf::cufft::get_handle_unique_ptr();

	std::size_t work_size;
	cufftMakePlan3d(*cufft_handle.get(), fft_dim[0], fft_dim[1], fft_dim[2], CUFFT_R2C, &work_size);

	CUTF_CHECK_ERROR(cufftXtExec(*cufft_handle.get(), device_data.get(), device_data.get(), CUFFT_FORWARD));
}
