#include <iostream>
#include <random>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

using compute_t = float;
constexpr std::size_t M = 1 << 5;
constexpr std::size_t N = 1 << 4;

template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * m + i]);
			if(val < 0.0f) {
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}


int main(){
	auto hA = cutf::memory::get_host_unique_ptr<compute_t>(M * N);
	auto hTAU = cutf::memory::get_host_unique_ptr<compute_t>(M * N);
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(M * N);
	auto dTAU = cutf::memory::get_device_unique_ptr<compute_t>(M * N);

	auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < M * N; i++){
		hA.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}
	print_matrix(hA.get(), M, N, "A");

	cutf::memory::copy(dA.get(), hA.get(), M * N);
	auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

	int Lwork_geqrf, Lwork_orgqr;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
			*cusolver.get(), M, N,
			dA.get(), M, &Lwork_geqrf));
	CUTF_CHECK_ERROR(cutf::cusolver::dn::orgqr_buffer_size(
			*cusolver.get(), M, N, N,
			dA.get(), M, dTAU.get(), &Lwork_orgqr));
	std::cout<<"Buffer size : "<<(sizeof(compute_t) * (Lwork_geqrf + Lwork_orgqr))<<"B"<<std::endl;

	auto dBuffer_geqrf = cutf::memory::get_device_unique_ptr<compute_t>(Lwork_geqrf);
	auto dBuffer_orgqr = cutf::memory::get_device_unique_ptr<compute_t>(Lwork_orgqr);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
				*cusolver.get(), M, N,
				dA.get(), M, dTAU.get(), dBuffer_geqrf.get(),
				Lwork_geqrf, dInfo.get()
				));

	int hInfo;
	cutf::memory::copy(&hInfo, dInfo.get(), 1);
	std::cout<<"geqrf info : "<<hInfo<<std::endl;

	cutf::memory::copy(hA.get(), dA.get(), M * N);
	print_matrix(hA.get(), M, N, "R");

	CUTF_CHECK_ERROR(cutf::cusolver::dn::orgqr(
				*cusolver.get(), M, N, N,
				dA.get(), M,
				dTAU.get(), dBuffer_orgqr.get(), Lwork_orgqr,
				dInfo.get()
				));
	cutf::memory::copy(&hInfo, dInfo.get(), 1);
	std::cout<<"orgqr info : "<<hInfo<<std::endl;
}
