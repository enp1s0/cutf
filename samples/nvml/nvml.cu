#include <cutf/nvml.hpp>
#include <iostream>

int main(){
	cutf::error::check(nvmlInit(), __FILE__, __LINE__, __func__);
	nvmlDevice_t device;
	cutf::error::check(nvmlDeviceGetHandleByIndex(0, &device), __FILE__, __LINE__, __func__);

	unsigned int temperature;
	cutf::error::check(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature), __FILE__, __LINE__, __func__);

	std::cout<<"GPU Temperature : "<<temperature<<" C"<<std::endl;

	cutf::error::check(nvmlShutdown(), __FILE__, __LINE__, __func__);
}
