#include <iostream>
#include <random>
#include <typeinfo>
#include <cutf/cusolver.hpp>
#include <cutf/cublas.hpp>
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/debug/matrix.hpp>

template <class T>
__global__ void cut_R_kernel(
    T* const dst_ptr,
    const T* const src_ptr, const std::size_t lds,
    const std::size_t n
    ) {
  const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
  const auto im = tid % n;
  const auto in = tid / n;
  if (im > in) {
    return;
  }

  dst_ptr[im + in * n] = src_ptr[im + lds * in];
}

template <class T>
void cut_R(
    T* const dst_ptr,
    const T* const src_ptr, const std::size_t lds,
    const std::size_t n
    ) {
  const auto num_threads = n * n;
  const auto block_size = 256;
  const auto grid_size = (num_threads + block_size - 1) / block_size;

  cut_R_kernel<<<grid_size, block_size>>>(
      dst_ptr,
      src_ptr, lds,
      n
      );
}

template <class T>
void eval(const std::size_t M, const std::size_t N) {
  auto hA = cutf::memory::get_host_unique_ptr<T>(M * N);
  auto hTAU = cutf::memory::get_host_unique_ptr<T>(M * N);
  auto dA = cutf::memory::get_device_unique_ptr<T>(M * N);
  auto dA_ref = cutf::memory::get_device_unique_ptr<T>(M * N);
  auto dR = cutf::memory::get_device_unique_ptr<T>(N * N);
  auto dTAU = cutf::memory::get_device_unique_ptr<T>(M * N);

  auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::mt19937 mt(std::random_device{}());

  for(auto i = decltype(N)(0); i < M * N; i++){
    hA.get()[i] = cutf::type::cast<T>(dist(mt));
  }

  cutf::memory::copy(dA.get(), hA.get(), M * N);
  cutf::memory::copy(dA_ref.get(), hA.get(), M * N);
  auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

  int Lwork_geqrf, Lwork_orgqr;
  CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
          *cusolver.get(), M, N,
          dA.get(), M, &Lwork_geqrf));
  CUTF_CHECK_ERROR(cutf::cusolver::dn::orgqr_buffer_size(
          *cusolver.get(), M, N, N,
          dA.get(), M, dTAU.get(), &Lwork_orgqr));

  auto dBuffer_geqrf = cutf::memory::get_device_unique_ptr<T>(Lwork_geqrf);
  auto dBuffer_orgqr = cutf::memory::get_device_unique_ptr<T>(Lwork_orgqr);


  CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
          *cusolver.get(), M, N,
          dA.get(), M, dTAU.get(), dBuffer_geqrf.get(),
          Lwork_geqrf, dInfo.get()
          ));
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  cut_R(dR.get(), dA.get(), M, N);
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  int hInfo;
  cutf::memory::copy(&hInfo, dInfo.get(), 1);

  cutf::memory::copy(hA.get(), dA.get(), M * N);

  CUTF_CHECK_ERROR(cutf::cusolver::dn::orgqr(
          *cusolver.get(), M, N, N,
          dA.get(), M,
          dTAU.get(), dBuffer_orgqr.get(), Lwork_orgqr,
          dInfo.get()
          ));
  cutf::memory::copy(&hInfo, dInfo.get(), 1);

  // Eval
  auto cublas_handle = cutf::cublas::get_cublas_unique_ptr();
  T diff_norm2, base_norm2;
  CUTF_CHECK_ERROR(cutf::cublas::dot(
          *cublas_handle.get(),
          M * N,
          dA_ref.get(), 1,
          dA_ref.get(), 1,
          &base_norm2
          ));
  const T p_one = 1, m_one = -1;
  CUTF_CHECK_ERROR(cutf::cublas::gemm(
          *cublas_handle.get(),
          CUBLAS_OP_N,
          CUBLAS_OP_N,
          M, N, N,
          &p_one,
          dA.get(), M,
          dR.get(), N,
          &m_one,
          dA_ref.get(), M
          ));
  CUTF_CHECK_ERROR(cutf::cublas::dot(
          *cublas_handle.get(),
          M * N,
          dA_ref.get(), 1,
          dA_ref.get(), 1,
          &diff_norm2
          ));
  CUTF_CHECK_ERROR(cudaDeviceSynchronize());
  const auto relative_residual = std::sqrt(diff_norm2 / base_norm2);
  std::printf("[M = %lu, N = %lu, dtype = %s] error = %e\n", M, N, typeid(T).name(), relative_residual);
}

int main() {
  for (std::size_t log_i = 5; log_i < 14; log_i += 2) {
    const auto N = 1lu << log_i;
    eval<float >(N, N);
    eval<double>(N, N);
  }
}
