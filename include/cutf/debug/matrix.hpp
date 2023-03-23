#ifndef __CUTF_DEBUG_MATRIX_HPP__
#define __CUTF_DEBUG_MATRIX_HPP__
#include <cstddef>
#include <stdio.h>
#include "../type.hpp"
#include "../macro.hpp"
#include "../memory.hpp"
#include "fp.hpp"
#include <cuComplex.h>

namespace cutf {
namespace debug {
namespace print {

namespace detail {

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_fp(const T a) {
	printf("%+e", cutf::type::cast<double>(a));
}

template <>
CUTF_DEVICE_HOST_FUNC inline void print_fp<cuComplex>(const cuComplex a) {
	printf("%+e%+ei", cutf::type::cast<double>(a.x), cutf::type::cast<double>(a.y));
}
template <>
CUTF_DEVICE_HOST_FUNC inline void print_fp<cuDoubleComplex>(const cuDoubleComplex a) {
	printf("%+e%+ei", cutf::type::cast<double>(a.x), cutf::type::cast<double>(a.y));
}
} // namespace detail

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(std::size_t i = 0; i < m; i++) {
		for(std::size_t j = 0; j < n; j++) {
			detail::print_fp(ptr[j * ldm + i]);
			printf(" ");
		}
		printf("\n");
	}
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	print_matrix(ptr, m, n, m, name);
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix_hex(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(std::size_t i = 0; i < m; i++) {
		for(std::size_t j = 0; j < n; j++) {
			const auto val = ptr[j * ldm + i];
			print::print_hex(val, false);
			printf(" ");
		}
		printf("\n");
	}
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix_hex(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	print_matrix_hex(ptr, m, n, m, name);
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_numpy_matrix(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	if(name != nullptr) printf("%s = ", name);
	printf("[");
	for(std::size_t i = 0; i < m; i++) {
		printf("[");
		for(std::size_t j = 0; j < n; j++) {
			detail::print_fp(ptr[j * ldm + i]);
			if (j + 1 < n)
				printf(",");
		}
		printf("]\n");
		if (i + 1 < m)
			printf(",");
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
inline void print_matrix_hex_from_host(const T* const ptr, const std::size_t m, const std::size_t n, const std::size_t ldm, const char* const name = nullptr) {
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	auto host_uptr = cutf::memory::get_host_unique_ptr<T>(ldm * n);
	cutf::memory::copy(host_uptr.get(), ptr, ldm * n);

	print_matrix_hex(host_uptr.get(), m, n, ldm, name);
}

template <class T>
inline void print_matrix_hex_from_host(const T* const ptr, const std::size_t m, const std::size_t n, const char* const name = nullptr) {
	CUTF_CHECK_ERROR(cudaDeviceSynchronize());
	auto host_uptr = cutf::memory::get_host_unique_ptr<T>(m * n);
	cutf::memory::copy(host_uptr.get(), ptr, m * n);

	print_matrix_hex(host_uptr.get(), m, n, name);
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
