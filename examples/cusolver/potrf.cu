#include <iostream>
#include <random>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/debug/matrix.hpp>

using compute_t = float;
constexpr std::size_t N = 1 << 6;

int main(){
	auto hA = cutf::memory::get_host_unique_ptr<compute_t>(N * N);
	auto hT = cutf::memory::get_host_unique_ptr<compute_t>(N * N);
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(N * N);

	auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < N * N; i++){
		hT.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}

#pragma omp parallel for collapse(2)
	for(auto i = decltype(N)(0); i < N; i++){
		for(auto j = decltype(N)(0); j < N; j++){
			compute_t c = 0;
			for(auto k = decltype(N)(0); k < N; k++){
				c += hT.get()[k * N + i] * hT.get()[k + j * N];
			}
			hA.get()[i + j * N] = c;
		}
	}

	cutf::memory::copy(dA.get(), hA.get(), N * N);
	auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

	int Lwork;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::potrf_buffer_size(*cusolver.get(), CUBLAS_FILL_MODE_FULL, N, dA.get(), N, &Lwork));
	std::cout<<"Buffer size : "<<(sizeof(compute_t) * (Lwork))<<"B"<<std::endl;

	auto dLwork_buffer = cutf::memory::get_device_unique_ptr<compute_t>(Lwork);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::potrf(
				*cusolver.get(),
				CUBLAS_FILL_MODE_FULL,
				N,
				dA.get(), N,
				dLwork_buffer.get(),
				Lwork,
				dInfo.get()
				));

	int hInfo;
	cutf::memory::copy(&hInfo, dInfo.get(), 1);
	std::cout<<"gesvd info : "<<hInfo<<std::endl;

	//cutf::debug::print::print_matrix(hA.get(), M, N, "A");
	//cutf::debug::print::print_matrix(hS.get(), 1, num_s, "S");
	//cutf::debug::print::print_matrix(hU.get(), M, num_s, "U");
	//cutf::debug::print::print_matrix(hVT.get(), num_s, N, "VT");
}
