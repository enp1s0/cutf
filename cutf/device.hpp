#ifndef __CUTF_DEVICE_HPP__
#define __CUTF_DEVICE_HPP__

#include <functional>
#include <vector>
#include "cuda.hpp"

namespace cutf{
namespace device{
inline std::vector<cudaDeviceProp> get_properties_vector(){
	int n;
	cudaGetDeviceCount(&n);
	std::vector<cudaDeviceProp> properties_vector;
	for(int i = 0; i < n; i++){
		cudaDeviceProp property;
		cudaGetDeviceProperties(&property, i);
		properties_vector.push_back(property);
	}

	return properties_vector;
}

inline void use_device(const int device_id, const std::function<void(void)> function){
	int current_device_id;
	cutf::error::check(cudaGetDevice(&current_device_id), __FILE__, __LINE__, __func__);
	cutf::error::check(cudaSetDevice(device_id), __FILE__, __LINE__, __func__);
	function();
	cutf::error::check(cudaSetDevice(current_device_id), __FILE__, __LINE__, __func__);
}
} // device
} // cutf

#endif // __CUTF_DEVICE_HPP__
