#ifndef __CUTF_MACRO_HPP__
#define __CUTF_MACRO_HPP__

#if !defined(CUTF_DEVICE_HOST_FUNC)
#define CUTF_DEVICE_HOST_FUNC __device__ __host__
#elif !defined(CUTF_DEVICE_HOST_FUNC)
#define CUTF_DEVICE_HOST_FUNC
#endif

#if !defined(CUTF_DEVICE_FUNC)
#define CUTF_DEVICE_FUNC __device__
#elif !defined(CUTF_DEVICE_FUNC)
#define CUTF_DEVICE_FUNC
#endif

// This macro prevents a warning "Unused variable"
#define CUTF_UNUSED(a) do {(void)(a);} while (0)

#endif // __CUTF_MACRO_HPP__
