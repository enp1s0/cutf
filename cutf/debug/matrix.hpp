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
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(std::size_t i = 0; i < m; i++) {
		for(std::size_t j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * m + i]);
			if(val == 0.0f) {
				printf(" %.5f ", 0.0);
			}else if (val < 0.0){
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}

template <class T>
CUTF_DEVICE_HOST_FUNC inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, std::size_t ldm, const char *name = nullptr) {
	if(name != nullptr) printf("%s = \n", name);
	for(std::size_t i = 0; i < m; i++) {
		for(std::size_t j = 0; j < n; j++) {
			const auto val = cutf::type::cast<float>(ptr[j * ldm + i]);
			if(val == 0.0f) {
				printf(" %.5f ", 0.0);
			}else if (val < 0.0){
				printf("%.5f ", val);
			}else{
				printf(" %.5f ", val);
			}
		}
		printf("\n");
	}
}
} // namespace print
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
