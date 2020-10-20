#ifndef __CUTF_DEBUG_RTPE_HPP__
#define __CUTF_DEBUG_RTPE_HPP__
#include "../macro.hpp"

namespace cutf {
namespace debug {
namespace type {

// name string
template <class T>
CUTF_DEVICE_HOST_FUNC inline const char* get_type_name();
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<double >() {return "double";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<float  >() {return "float";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<__half >() {return "half/(u)int16_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<__half2>() {return "half2";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<long long>() {return "uint64_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<uint64_t>() {return "uint64_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<uint32_t>() {return "uint32_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<uint16_t>() {return "uint8_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<int64_t>() {return "int64_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<int32_t>() {return "int32_t";}
template <> CUTF_DEVICE_HOST_FUNC inline const char* get_type_name<int16_t>() {return "int8_t";}

} // namespace type
} // namespace debug
} // namespace cutf
#endif
