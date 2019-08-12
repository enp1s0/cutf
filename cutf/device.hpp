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

inline cudaError_t use_device(const int device_id, const std::function<void(void)> function){
	int current_device_id;
	cudaError_t result;
	result = cudaGetDevice(&current_device_id);
	if(result != cudaSuccess) {return result;}
	result = cudaSetDevice(device_id);
	if(result != cudaSuccess) {return result;}
	function();
	result = cudaSetDevice(current_device_id);
	if(result != cudaSuccess) {return result;}

	return cudaSuccess;
}
} // device
} // cutf

#endif // __CUTF_DEVICE_HPP__
