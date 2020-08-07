#ifndef __CUTF_MACRO_HPP__
#define __CUTF_MACRO_HPP__
#if !defined(CUTF_DEVICE_HOST_FUNC) && defined(__CUDA_ARCH__)
#define CUTF_DEVICE_HOST_FUNC __device__ __host__
#elif !defined(CUTF_DEVICE_HOST_FUNC)
#define CUTF_DEVICE_HOST_FUNC
#endif

#if !defined(CUTF_DEVICE_FUNC) && defined(__CUDA_ARCH__)
#define CUTF_DEVICE_FUNC __device__
#elif !defined(CUTF_DEVICE_FUNC)
#define CUTF_DEVICE_FUNC
#endif

#endif
