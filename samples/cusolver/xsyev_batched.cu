#include <iostream>
#include <random>
#include <cutf/cusolver.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/debug/matrix.hpp>

using compute_t = float;
constexpr std::size_t M = 1 << 10;
constexpr std::size_t batch_size = 1 << 10;

int main(){
  auto hA = cutf::memory::get_host_unique_ptr<compute_t>(M * M);
  auto dA = cutf::memory::get_device_unique_ptr<compute_t>(M * M);
  auto dW = cutf::memory::get_device_unique_ptr<compute_t>(M * batch_size);

  auto dInfo = cutf::memory::get_device_unique_ptr<int>(batch_size);

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 mt(std::random_device{}());

  for(auto i = decltype(M)(0); i < M * M; i++){
    hA.get()[i] = cutf::type::cast<compute_t>(dist(mt));
  }

  cutf::memory::copy(dA.get(), hA.get(), M * M);
  auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();
  auto params_uptr = cutf::cusolver::dn::get_params_unique_ptr();

  std::size_t device_w_size, host_w_size;
  CUTF_CHECK_ERROR(cusolverDnXsyevBatched_bufferSize(
          *cusolver.get(),
          *params_uptr.get(),
          CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER,
          M,
          cutf::type::get_data_type<compute_t>(),
          dA.get(),
          M,
          cutf::type::get_data_type<compute_t>(),
          dW.get(),
          cutf::type::get_data_type<compute_t>(),
          &device_w_size,
          &host_w_size,
          batch_size));
  std::cout << "Buffer size : " << device_w_size << "B for device and " << host_w_size << "B for host" << std::endl;

  auto device_work_buffer = cutf::memory::get_device_unique_ptr<std::uint8_t>(device_w_size);
  auto host_work_buffer = cutf::memory::get_device_unique_ptr<std::uint8_t>(host_w_size);

  CUTF_CHECK_ERROR(cusolverDnXsyevBatched(
          *cusolver.get(),
          *params_uptr.get(),
          CUSOLVER_EIG_MODE_VECTOR,
          CUBLAS_FILL_MODE_UPPER,
          M,
          cutf::type::get_data_type<compute_t>(),
          dA.get(),
          M,
          cutf::type::get_data_type<compute_t>(),
          dW.get(),
          cutf::type::get_data_type<compute_t>(),
          device_work_buffer.get(),
          device_w_size,
          host_work_buffer.get(),
          host_w_size,
          dInfo.get(),
          batch_size));

  auto hInfo_uptr = cutf::memory::get_host_unique_ptr<int>(batch_size);
  cutf::memory::copy(hInfo_uptr.get(), dInfo.get(), batch_size);

  /*
  cutf::memory::copy(hS.get(), dS.get(), num_s);
  cutf::memory::copy(hU.get(), dU.get(), M * num_s);
  cutf::memory::copy(hVT.get(), dVT.get(), N * num_s);

  double base_norm2 = 0;
  double diff_norm2 = 0;

#pragma omp parallel for collapse(2) reduction(+: base_norm2) reduction(+: diff_norm2)
  for (std::size_t i = 0; i < M; i++) {
    for (std::size_t j = 0; j < N; j++) {
      double c = 0;
      for (std::size_t s = 0; s < num_s; s++) {
        c += hU.get()[i + s * M] * hS.get()[s] * hVT.get()[s + j * num_s];
      }
      base_norm2 += c * c;
      const auto diff = c - hA.get()[i + j * M];
      diff_norm2 += diff * diff;
    }
  }
  const auto relative_error = std::sqrt(diff_norm2 / base_norm2);
  std::printf("relative error = %e\n", relative_error);
  */
}
