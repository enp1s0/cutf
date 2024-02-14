#include <cutf/curand.hpp>
#include <cutf/memory.hpp>

constexpr std::size_t N = 1lu << 10;

int main() {
	unsigned long long seed = 10;
	auto cugen = cutf::curand::get_curand_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	auto cugen_host = cutf::curand::get_curand_host_unique_ptr(CURAND_RNG_PSEUDO_MT19937);
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen.get(), seed));
	CUTF_CHECK_ERROR(curandSetPseudoRandomGeneratorSeed(*cugen_host.get(), seed));

	auto dA = cutf::memory::get_device_unique_ptr<float>(N);
	auto hA = cutf::memory::get_host_unique_ptr<float>(N);
	auto hA_host = cutf::memory::get_host_unique_ptr<float>(N);

	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen.get(), dA.get(), N));
	cutf::memory::copy(hA.get(), dA.get(), N);
	CUTF_CHECK_ERROR(cutf::curand::generate_uniform(*cugen_host.get(), hA_host.get(), N));

	double error_sum = 0.0;
	for (std::size_t i = 0; i < N; i++) {
		const auto diff = hA.get()[i] - hA_host.get()[i];
		error_sum = std::abs(diff);
	}

	std::printf("error mean : %e\n", error_sum / N);
}
