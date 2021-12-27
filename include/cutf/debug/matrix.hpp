#ifndef __CUTF_DEBUG_MATRIX_HPP__
#define __CUTF_DEBUG_MATRIX_HPP__
#include <cstddef>
#include <stdio.h>
#include "../type.hpp"
#include "../macro.hpp"
#include "../memory.hpp"

namespace cutf {
namespace debug {
namespace print {

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(std::size_t i = 0; i < m; i++) {
		for(std::size_t j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			printf("%+e ", val);
		}
		printf("\n");
	}
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	print_matrix(ptr, m, n, m, name);
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_numpy_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	if(name != nullptr) printf("%s = ", name);
	printf("[");
	for(std::size_t i = 0; i < m; i++) {
		printf("[");
		for(std::size_t j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			printf("%+e,", val);
		}
		printf("],\n");
	}
	printf("]\n");
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_numpy_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	print_numpy_matrix(ptr, m, n, m, name);
}

// For device moery
template <class T>
inline void print_matrix_from_host(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	auto host_uptr = cutf::memory::get_host_unique_ptr<T>(ldm * n);
	cutf::memory::copy(host_uptr.get(), ptr, ldm * n);

	print_matrix(host_uptr.get(), m, n, ldm, name);
}

template <class T>
inline void print_matrix_from_host(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	auto host_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
	cutf::memory::copy(host_uptr.get(), ptr, m * n);

	print_matrix(host_uptr.get(), m, n, name);
}

template <class T>
inline void print_numpy_matrix_from_host(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	auto host_uptr = cutf::memory::get_host_unique_ptr<T>(ldm * n);
	cutf::memory::copy(host_uptr.get(), ptr, ldm * n);

	print_numpy_matrix(host_uptr.get(), m, n, ldm, name);
}

template <class T>
inline void print_numpy_matrix_from_host(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	auto host_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
	cutf::memory::copy(host_uptr.get(), ptr, m * n);

	print_numpy_matrix(host_uptr.get(), m, n, name);
}
} // namespace print
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
