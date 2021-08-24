# cuSOLVER Functions

## Example
```cpp
#include <cutf/type.hpp>
#include <cutf/memory.hpp>
#include <cutf/cusoler.hpp>
constexpr std::size_t N = 1<<10;
constexpr std::size_t M = 1<<10;

using T = half;
int main(){
	auto dA = cutf::memory::get_device_unique_ptr<compute_t>(M * N);
	auto dTAU = cutf::memory::get_device_unique_ptr<compute_t>(M * N);

	auto dInfo = cutf::memory::get_device_unique_ptr<int>(1);

	auto cusolver = cutf::cusolver::dn::get_handle_unique_ptr();

	int Lwork_geqrf, Lwork_orgqr;
	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf_buffer_size(
			*cusolver.get(), M, N,
			dA.get(), M, &Lwork_geqrf));
	CUTF_CHECK_ERROR(cutf::cusolver::dn::gqr_buffer_size(
			*cusolver.get(), M, N, N,
			dA.get(), M, dTAU.get(), &Lwork_orgqr));

	auto dBuffer_geqrf = cutf::memory::get_device_unique_ptr<compute_t>(Lwork_geqrf);
	auto dBuffer_orgqr = cutf::memory::get_device_unique_ptr<compute_t>(Lwork_orgqr);

	CUTF_CHECK_ERROR(cutf::cusolver::dn::geqrf(
				*cusolver.get(), M, N,
				dA.get(), M, dTAU.get(), dBuffer_geqrf.get(),
				Lwork_geqrf, dInfo.get()
				));

}
```

## Namespace
- `cutf::dn::` : Functions for dense matrices
- `cutf::sp::` : Functions for sparce matrices

## Supported functions
### helper

- [x] `dn::get_handle_unique_ptr`
- [x] `sp::get_handle_unique_ptr`
- [x] `dn::get_params_unique_ptr`

### Dense linear solver (in `dn` namespace)

- [x] `potrf`
- [x] `potrs`
- [x] `potri`
- [x] `getrf`
- [x] `getrs`
- [x] `geqrf`
- [x] `ormqr`
- [x] `orgqr`
- [x] `sytrf`
- [ ] `potrfBatched`
- [ ] `potrsBatched`

### Dense eigenvalue solver (in `dn` namespace)

- [x] `gebrd`
- [x] `orgbr`
- [x] `sytrd`
- [x] `ormtr`
- [x] `orgtr`
- [x] `gesvd`
- [x] `gesvdj`
- [ ] `gesvdjBatched`
- [ ] `gesvdaStridedBatched`
- [ ] `syevd`
- [ ] `syevdx`
- [ ] `sygvd`
- [ ] `sygvdx`
- [ ] `syevj`
- [ ] `sygvj`
- [ ] `syevjBatched`


## Reference
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cusolver/index.html)
