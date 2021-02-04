#ifndef __CUTF_DEBUG_MATRIX_HPP__
#define __CUTF_DEBUG_MATRIX_HPP__
#include <cstddef>
#include <stdio.h>
#include "../type.hpp"
#include "../macro.hpp"

namespace cutf {
namespace debug {
namespace print {

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, std::size_t ldm, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(std::size_t i = 0; i < m; i++) {
		for(std::size_t j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			if(val == 0.0f) {
				printf(" %e ", 0.0);
			}else if (val < 0.0){
				printf("%e ", val);
			}else{
				printf(" %e ", val);
			}
		}
		printf("\n");
	}
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	print_matrix(ptr, m, n, m, name);
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_numpy_matrix(const T* const ptr, std::size_t m, std::size_t n, std::size_t ldm, const char *name = nullptr) {
	if(name != nullptr) printf("%s = ", name);
	printf("[");
	for(std::size_t i = 0; i < m; i++) {
		printf("[");
		for(std::size_t j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			if(val == 0.0f) {
				printf(" %e,", 0.0);
			}else if (val < 0.0){
				printf("%e,", val);
			}else{
				printf(" %e,", val);
			}
		}
		printf("],\n");
	}
	printf("]\n");
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_numpy_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	print_numpy_matrix(ptr, m, n, m, name);
}
} // namespace print
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
