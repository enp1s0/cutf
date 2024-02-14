#include <iostream>
#include <chrono>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/curand.hpp>

template <class compute_t>
void rsvd_test(const std::size_t m, const std::size_t n, const std::size_t k, const std::size_t p, const std::size_t niter) {
	try {
		auto d_a_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * n);
		auto d_s_uptr = cutf::memory::get_device_unique_ptr<compute_t>(std::min(m, n));
		auto d_u_uptr = cutf::memory::get_device_unique_ptr<compute_t>(m * k);
		auto d_v_uptr = cutf::memory::get_device_unique_ptr<compute_t>(k * n);
		auto h_a_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * n);
		auto h_s_uptr = cutf::memory::get_host_unique_ptr<compute_t>(std::min(m, n));
		auto h_u_uptr = cutf::memory::get_host_unique_ptr<compute_t>(m * k);
		auto h_v_uptr = cutf::memory::get_host_unique_ptr<compute_t>(k * n);

		// init A
		auto curand_generator_uptr = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_DEFAULT);
		CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*curand_generator_uptr.get(), 0));
		CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*curand_generator_uptr.get(), d_a_uptr.get(), m * n / sizeof(compute_t)));

		// prepare cusolver
		auto cusolver_handle = cutf::cusolver::dn::get_handle_unique_ptr();
		auto cusolver_params = cutf::cusolver::dn::get_params_unique_ptr();
		CUTF_CHECK_ERROR(cusolverDnSetAdvOptions(*cusolver_params.get(), CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

		std::size_t working_memory_device_size;
		std::size_t working_memory_host_size;
		CUTF_CHECK_ERROR(cusolverDnXgesvdr_bufferSize(
					*cusolver_handle.get(),
					*cusolver_params.get(),
					'S', 'S',
					m, n, k, p,
					niter,
					cutf::type::get_data_type<compute_t>(),
					d_a_uptr.get(), m,
					cutf::type::get_data_type<compute_t>(),
					d_s_uptr.get(),
					cutf::type::get_data_type<compute_t>(),
					d_u_uptr.get(), m,
					cutf::type::get_data_type<compute_t>(),
					d_v_uptr.get(), n,
					cutf::type::get_data_type<compute_t>(),
					&working_memory_device_size,
					&working_memory_host_size
					));

		auto working_memory_host_uptr = cutf::memory::get_host_unique_ptr<uint8_t>(working_memory_host_size);
		auto working_memory_device_uptr = cutf::memory::get_device_unique_ptr<uint8_t>(working_memory_device_size);
		auto devInfo_uptr = cutf::memory::get_device_unique_ptr<int>(1);

		cudaDeviceSynchronize();
		const auto start_clock = std::chrono::system_clock::now();
		CUTF_CHECK_ERROR(cusolverDnXgesvdr(
					*cusolver_handle.get(),
					*cusolver_params.get(),
					'S', 'S',
					m, n, k, p,
					niter,
					cutf::type::get_data_type<compute_t>(),
					d_a_uptr.get(), m,
					cutf::type::get_data_type<compute_t>(),
					d_s_uptr.get(),
					cutf::type::get_data_type<compute_t>(),
					d_u_uptr.get(), m,
					cutf::type::get_data_type<compute_t>(),
					d_v_uptr.get(), n,
					cutf::type::get_data_type<compute_t>(),
					working_memory_device_uptr.get(),
					working_memory_device_size,
					working_memory_host_uptr.get(),
					working_memory_host_size,
					devInfo_uptr.get()
					));
		cudaDeviceSynchronize();
		const auto end_clock = std::chrono::system_clock::now();

		const auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;

		std::printf("%lu,%lu,%lu,%lu,%lu,%lu,%lu,%e\n",
				m, n, k, p,
				niter,
				working_memory_device_size, working_memory_host_size,
				elapsed_time
				);
		std::fflush(stdout);
	} catch(const std::exception&) {}
}

int main() {
	rsvd_test<float >(100, 100, 50, 10, 10);
	rsvd_test<double>(100, 100, 50, 10, 10);
}
