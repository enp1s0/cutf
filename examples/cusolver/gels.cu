#include <iostream>
#include <random>
#include <typeinfo>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>

using compute_t = float;
constexpr std::size_t M = 1 << 12;
constexpr std::size_t N = 1 << 12;

template <class compute_t, class lowest_t>
void test(){
  std::printf("# GELS test<main=%s, lowest=%s>\n", typeid(compute_t).name(), typeid(lowest_t).name());
	auto A_uptr = cutf::memory::get_managed_unique_ptr<compute_t>(M * N);
	auto x_uptr = cutf::memory::get_managed_unique_ptr<compute_t>(N);
	auto b_uptr = cutf::memory::get_managed_unique_ptr<compute_t>(M);

	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	std::mt19937 mt(std::random_device{}());

	for(auto i = decltype(N)(0); i < M * N; i++){
		A_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}
	for(auto i = decltype(N)(0); i < N; i++){
		x_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}
	for(auto i = decltype(N)(0); i < M; i++){
		b_uptr.get()[i] = cutf::type::cast<compute_t>(dist(mt));
	}

  std::printf("A : %lu x %lu\n", M, N);

	auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

  std::size_t lwork_bytes;
  CUTF_CHECK_ERROR(cutf::cusolver::dn::gels_buffer_size<lowest_t>(
        *cusolver.get(),
        M,
        N,
        1,
        A_uptr.get(), M,
        b_uptr.get(), M,
        x_uptr.get(), N,
        nullptr,
        &lwork_bytes
        )
      );

  auto lwork_uptr = cutf::memory::get_device_unique_ptr<std::uint8_t>(lwork_bytes);
  auto info_uptr = cutf::memory::get_managed_unique_ptr<int>(1);

  int niters;
  CUTF_CHECK_ERROR(cutf::cusolver::dn::gels<lowest_t>(
        *cusolver.get(),
        M,
        N,
        1,
        A_uptr.get(), M,
        b_uptr.get(), M,
        x_uptr.get(), N,
        lwork_uptr.get(),
        lwork_bytes,
        &niters,
        info_uptr.get()
        )
      );

  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  std::printf("dinfo = %d\n", *info_uptr.get());
  std::printf("iter = %d\n", niters);

  double base_norm2 = 0;
  double diff_norm2 = 0;
  for (std::size_t i = 0; i < M; i++) {
    double c = 0;
    for (std::size_t j = 0; j < N; j++) {
      c += static_cast<double>(A_uptr.get()[i + j * M]) * x_uptr.get()[j];
    }
    const auto d = b_uptr.get()[i];
    const auto diff = c - d;
    base_norm2 += d * d;
    diff_norm2 += diff * diff;
  }
  const auto relative_error = std::sqrt(diff_norm2 / base_norm2);
  std::printf("relative error = %e\n", relative_error);
}

int main() {
  test<float, float>();
  test<float, half>();
  test<float, nv_bfloat16>();
}
