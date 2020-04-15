#ifndef __DEBUG_MATRIX_HPP__
#define __DEBUG_MATRIX_HPP__
#include <cstddef>
#include <stdio.h>
#include "../type.hpp"

namespace cutf {
namespace debug {
namespace matrix {

template <class T>
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, const char *name = nullptr) {
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
__device__ __host__ inline void print_matrix(const T* const ptr, std::size_t m, std::size_t n, std::size_t ldm, const char *name = nullptr) {
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
} // namespace matrix
} // namespace debug
} // namespace cutf

#endif /* end of include guard */
