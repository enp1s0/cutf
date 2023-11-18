#include <cutf/cufft.hpp>
#include <cutf/memory.hpp>
#include <cutf/curand.hpp>
#include <iostream>
#include <vector>
#include <random>

std::string vec_str(const std::vector<std::size_t>& v) {
  std::string str = "";
  for (std::size_t i = 0; i < v.size(); i++) {
    str += std::to_string(v[i]);
    if (i != v.size() - 1) {
      str += ",";
    }
  }
  return str;
}

template <class T>
struct real_t {
  using type = T;
  const static std::size_t num = 1;
};
template <>
struct real_t<cuComplex> {
  using type = float;
  const static std::size_t num = 2;
};
template <>
struct real_t<cuDoubleComplex> {
  using type = float;
  const static std::size_t num = 2;
};

template <class T>
typename real_t<T>::type real_value(const T v) {return v;}
template <>
typename real_t<cuComplex>::type real_value(const cuComplex v) {return v.x;}
template <>
typename real_t<cuDoubleComplex>::type real_value(const cuDoubleComplex v) {return v.x;}

template <class T>
void rand_init(T* const ptr, const std::size_t len) {
	unsigned long long seed = 10;
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_DEFAULT);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), reinterpret_cast<typename real_t<T>::type*>(ptr), len * real_t<T>::num));
}

template <class IN_T, class OUT_T>
void eval(const std::vector<std::size_t>& dims) {
  std::size_t num_elements = 1;
  for (const auto v : dims) {
    num_elements *= v;
  }

  auto in_uptr  = cutf::memory::get_managed_unique_ptr<IN_T >(num_elements);
  auto out_uptr = cutf::memory::get_managed_unique_ptr<OUT_T>(num_elements);
  auto ref_uptr = cutf::memory::get_managed_unique_ptr<OUT_T>(num_elements);

  rand_init(in_uptr.get(), num_elements);

  cufftHandle plan;
  const auto type = cutf::cufft::get_type<IN_T, OUT_T>();
  switch (dims.size()) {
    case 1:
      CUTF_HANDLE_ERROR(cufftPlan1d(&plan, dims[0], type, 1));
    case 2:
      CUTF_HANDLE_ERROR(cufftPlan2d(&plan, dims[0], dims[1], type));
      break;
    case 3:
      CUTF_HANDLE_ERROR(cufftPlan3d(&plan, dims[0], dims[1], dims[2], type));
      break;
    default:
      break;
  }

  CUTF_CHECK_ERROR(cufftXtExec(plan, in_uptr.get(), out_uptr.get(),CUFFT_FORWARD));
  CUTF_CHECK_ERROR(cufftXtExec(plan, out_uptr.get(), ref_uptr.get(),CUFFT_INVERSE));

  CUTF_CHECK_ERROR(cudaDeviceSynchronize());

  double max_error = 0;
  for (std::size_t i = 0; i < num_elements; i++) {
    max_error = std::max(static_cast<double>(real_value(in_uptr.get()[i])) - real_value(ref_uptr.get()[i]), max_error);
  }

  std::printf("dims=(%s), max_error=%e\n", vec_str(dims).c_str(), max_error);
}

int main() {
  for (const auto dims : std::vector<std::vector<std::size_t>>{{1024}, {256, 256}, {32, 32, 32}}) {
    eval<float    , cuComplex>(dims);
    eval<cuComplex, cuComplex>(dims);
    eval<cuComplex, float    >(dims);
    eval<double         , cuDoubleComplex>(dims);
    eval<cuDoubleComplex, cuDoubleComplex>(dims);
    eval<cuDoubleComplex, double         >(dims);
  }
}
