#include <iostream>
#include <cassert>
#include <cutf/type.hpp>

#define CHECK_TYPE(T, cuda_type) \
	static_assert(cutf::type::get_data_type<T>() == cuda_type, "Type unmatched : " #cuda_type)

#define TYPE_CAST_TEST(DST, SRC) \
{const SRC a = 1; const auto v = cutf::type::cast<DST>(a);CUTF_UNUSED(v);}

#ifdef __CUTF_FP8_EXIST__
#define TYPE_CAST_TEST_1(DST) \
	TYPE_CAST_TEST(DST, half           ) \
	TYPE_CAST_TEST(DST, __nv_bfloat16  ) \
	TYPE_CAST_TEST(DST, float          ) \
	TYPE_CAST_TEST(DST, double         ) \
	TYPE_CAST_TEST(DST, std::uint8_t   ) \
	TYPE_CAST_TEST(DST, std::int8_t    ) \
	TYPE_CAST_TEST(DST, std::uint16_t  ) \
	TYPE_CAST_TEST(DST, std::int16_t   ) \
	TYPE_CAST_TEST(DST, std::uint32_t  ) \
	TYPE_CAST_TEST(DST, std::int32_t   ) \
	TYPE_CAST_TEST(DST, std::uint64_t  ) \
	TYPE_CAST_TEST(DST, std::int64_t   ) \
	TYPE_CAST_TEST(DST, unsigned       ) \
	TYPE_CAST_TEST(DST, int            ) \
	TYPE_CAST_TEST(DST, __nv_fp8_e5m2  ) \
	TYPE_CAST_TEST(DST, __nv_fp8_e4m3  )

#define TYPE_CAST_TEST_2 \
	TYPE_CAST_TEST_1(half           ) \
	TYPE_CAST_TEST_1(__nv_bfloat16  ) \
	TYPE_CAST_TEST_1(float          ) \
	TYPE_CAST_TEST_1(double         ) \
	TYPE_CAST_TEST_1(std::uint8_t   ) \
	TYPE_CAST_TEST_1(std::int8_t    ) \
	TYPE_CAST_TEST_1(std::uint16_t  ) \
	TYPE_CAST_TEST_1(std::int16_t   ) \
	TYPE_CAST_TEST_1(std::uint32_t  ) \
	TYPE_CAST_TEST_1(std::int32_t   ) \
	TYPE_CAST_TEST_1(std::uint64_t  ) \
	TYPE_CAST_TEST_1(std::int64_t   ) \
	TYPE_CAST_TEST_1(unsigned       ) \
	TYPE_CAST_TEST_1(int            ) \
	TYPE_CAST_TEST_1(__nv_fp8_e5m2  ) \
	TYPE_CAST_TEST_1(__nv_fp8_e4m3  )
#else
#define TYPE_CAST_TEST_1(DST) \
	TYPE_CAST_TEST(DST, half           ) \
	TYPE_CAST_TEST(DST, __nv_bfloat16  ) \
	TYPE_CAST_TEST(DST, float          ) \
	TYPE_CAST_TEST(DST, double         ) \
	TYPE_CAST_TEST(DST, std::uint8_t   ) \
	TYPE_CAST_TEST(DST, std::int8_t    ) \
	TYPE_CAST_TEST(DST, std::uint16_t  ) \
	TYPE_CAST_TEST(DST, std::int16_t   ) \
	TYPE_CAST_TEST(DST, std::uint32_t  ) \
	TYPE_CAST_TEST(DST, std::int32_t   ) \
	TYPE_CAST_TEST(DST, std::uint64_t  ) \
	TYPE_CAST_TEST(DST, std::int64_t   ) \
	TYPE_CAST_TEST(DST, unsigned       ) \
	TYPE_CAST_TEST(DST, int            )

#define TYPE_CAST_TEST_2 \
	TYPE_CAST_TEST_1(half           ) \
	TYPE_CAST_TEST_1(__nv_bfloat16  ) \
	TYPE_CAST_TEST_1(float          ) \
	TYPE_CAST_TEST_1(double         ) \
	TYPE_CAST_TEST_1(std::uint8_t   ) \
	TYPE_CAST_TEST_1(std::int8_t    ) \
	TYPE_CAST_TEST_1(std::uint16_t  ) \
	TYPE_CAST_TEST_1(std::int16_t   ) \
	TYPE_CAST_TEST_1(std::uint32_t  ) \
	TYPE_CAST_TEST_1(std::int32_t   ) \
	TYPE_CAST_TEST_1(std::uint64_t  ) \
	TYPE_CAST_TEST_1(std::int64_t   ) \
	TYPE_CAST_TEST_1(unsigned       ) \
	TYPE_CAST_TEST_1(int            )
#endif

int main() {
	CHECK_TYPE(half           , CUDA_R_16F);
	CHECK_TYPE(__nv_bfloat16  , CUDA_R_16BF);
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

#ifdef __CUTF_FP8_EXIST__
	CHECK_TYPE(__nv_fp8_e5m2  , CUDA_R_8F_E5M2);
	CHECK_TYPE(__nv_fp8_e4m3  , CUDA_R_8F_E4M3);
#endif

  TYPE_CAST_TEST_2
}
