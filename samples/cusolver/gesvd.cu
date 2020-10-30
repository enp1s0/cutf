#include <iostream>
#include <random>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

using compute_t = float;
constexpr std::size_t M = 1 << 4;
constexpr std::size_t N = 1 << 5;

template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(int i = 0; i < m; i++) {
		for(int j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * m + i]);
			if(val == 0) {
				printf(" %.5f ", 0.0);
			}else if(val < 0) {
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
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(M * N);

	constexpr std::size_t num_s = std::min(M, N);
	auto dS = cutf::memory::get_device_unique_ptr<compute_t>(num_s);
	auto dU = cutf::memory::get_device_unique_ptr<compute_t>(M * M);
	auto dVT = cutf::memory::get_device_unique_ptr<compute_t>(N * N);
	auto hS = cutf::memory::get_host_unique_ptr<compute_t>(num_s);
	auto hU = cutf::memory::get_host_unique_ptr<compute_t>(num_s * M);
	auto hVT = cutf::memory::get_host_unique_ptr<compute_t>(num_s * N);

	auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < M * N; i++){
		hA.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}

	cutf::memory::copy(dA.get(), hA.get(), M * N);
	auto cusolver = cutf::cusolver::get_cusolver_dn_unique_ptr();

	int Lwork;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::gesvd_buffer_size<compute_t>(*cusolver.get(), M, N, &Lwork));
	std::cout<<"Buffer size : "<<(sizeof(compute_t) * (Lwork))<<"B"<<std::endl;

	auto dLwork_buffer = cutf::memory::get_device_unique_ptr<compute_t>(Lwork);
	auto dRwork_buffer = cutf::memory::get_device_unique_ptr<compute_t>(num_s - 1);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::gesvd(
				*cusolver.get(),
				'S', 'S',
				M, N,
				dA.get(), M,
				dS.get(),
				dU.get(), M,
				dVT.get(), num_s,
				dLwork_buffer.get(),
				Lwork,
				dRwork_buffer.get(),
				dInfo.get()
				));

	int hInfo;
	cutf::memory::copy(&hInfo, dInfo.get(), 1);
	std::cout<<"gesvd info : "<<hInfo<<std::endl;

	cutf::memory::copy(hS.get(), dS.get(), num_s);
	cutf::memory::copy(hU.get(), dU.get(), M * num_s);
	cutf::memory::copy(hVT.get(), dVT.get(), N * num_s);

	print_matrix(hA.get(), M, N, "A");
	print_matrix(hS.get(), 1, num_s, "S");
	print_matrix(hU.get(), M, num_s, "U");
	print_matrix(hVT.get(), num_s, N, "VT");
}
