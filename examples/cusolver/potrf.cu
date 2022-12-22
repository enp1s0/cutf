#include <iostream>
#include <random>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/debug/matrix.hpp>

using compute_t = float;
constexpr std::size_t default_N = 1 << 13;

int main(int argc, char** argv){
	std::size_t N = default_N;
	if (argc >= 2) {
		N = std::stoul(argv[1]);
	}

	auto hA = cutf::memory::get_host_unique_ptr<compute_t>(N * N);
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(N * N);

	auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < N * N; i++){
		hA.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}

	cutf::memory::copy(dA.get(), hA.get(), N * N);
	auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

	int Lwork;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::potrf_buffer_size(*cusolver.get(), CUBLAS_FILL_MODE_UPPER, N, dA.get(), N, &Lwork));
	std::cout<<"Buffer size : "<<(sizeof(compute_t) * (Lwork))<<"B"<<std::endl;

	auto dLwork_buffer = cutf::memory::get_device_unique_ptr<compute_t>(Lwork);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::potrf(
				*cusolver.get(),
				CUBLAS_FILL_MODE_UPPER,
				N,
				dA.get(), N,
				dLwork_buffer.get(),
				Lwork,
				dInfo.get()
				));

	int hInfo;
	cutf::memory::copy(&hInfo, dInfo.get(), 1);
	std::cout<<"gesvd info : "<<hInfo<<std::endl;
}
