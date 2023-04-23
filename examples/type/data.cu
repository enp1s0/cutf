#include <iostream>
#include <cassert>
#include <cutf/type.hpp>

#define CHECK_TYPE(T, cuda_type) \
	static_assert(cutf::type::get_data_type<T>() == cuda_type, "Type unmatched : " #cuda_type)

int main() {
	CHECK_TYPE(float          , CUDA_R_32F);
	CHECK_TYPE(double         , CUDA_R_64F);
	CHECK_TYPE(std::uint8_t   , CUDA_R_8U);
	CHECK_TYPE(std::int8_t    , CUDA_R_8I);
	CHECK_TYPE(std::uint16_t  , CUDA_R_16U);
	CHECK_TYPE(std::int16_t   , CUDA_R_16I);
	CHECK_TYPE(std::uint32_t  , CUDA_R_32U);
	CHECK_TYPE(std::int32_t   , CUDA_R_32I);
	CHECK_TYPE(std::uint64_t  , CUDA_R_64U);
	CHECK_TYPE(std::int64_t   , CUDA_R_64I);
	CHECK_TYPE(unsigned       , CUDA_R_32U);
	CHECK_TYPE(int            , CUDA_R_32I);
	CHECK_TYPE(cuComplex      , CUDA_C_32F);
	CHECK_TYPE(cuDoubleComplex, CUDA_C_64F);
}
