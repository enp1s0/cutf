#include <cutf/cutensor.hpp>
#include <cutf/error.hpp>

int main() {
	cutensorHandle_t handle;
	CUTF_CHECK_ERROR(cutensorInit(&handle));

	{const auto compute_type = cutf::cutensor::get_compute_type<half    >();}
	{const auto compute_type = cutf::cutensor::get_compute_type<float   >();}
	{const auto compute_type = cutf::cutensor::get_compute_type<double  >();}
	{const auto compute_type = cutf::cutensor::get_compute_type<uint32_t>();}
	{const auto compute_type = cutf::cutensor::get_compute_type<int32_t >();}
	{const auto compute_type = cutf::cutensor::get_compute_type<uint8_t >();}
	{const auto compute_type = cutf::cutensor::get_compute_type<int8_t  >();}
	{const auto compute_type = cutf::cutensor::get_compute_type<nvcuda::wmma::precision::tf32>();}
	{const auto compute_type = cutf::cutensor::get_compute_type<__nv_bfloat16>();}
}
