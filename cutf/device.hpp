#ifndef __CUTF_DEVICE_HPP__
#define __CUTF_DEVICE_HPP__

#include <vector>

namespace cutf{
namespace cuda{
namespace device{
inline std::vector<cudaDeviceProp> get_properies_vector(){
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
} // device
} // cuda
} // cutf

#endif // __CUTF_DEVICE_HPP__
