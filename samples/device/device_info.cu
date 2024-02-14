#include <iostream>
#include <cutf/device.hpp>

int main(){
	const auto device_properties = cutf::device::get_properties_vector();

	int device_id = 0;
	for(const auto property : device_properties){
		std::cout
			<<" Device ID   : "<<(device_id++)<<std::endl
			<<" Device Name : "<<property.name<<std::endl
			<<" Compute Cap : "<<property.major<<"."<<property.minor<<std::endl
			<<"#--"<<std::endl;
	}
}
