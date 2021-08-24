#ifndef __CUTF_EVENT_HPP__
#define __CUTF_EVENT_HPP__
#include <memory>
#include "cuda.hpp"

namespace cutf {
namespace event {
struct event_deleter{
	void operator()(cudaEvent_t* event){
		cutf::error::check(cudaEventDestroy(*event), __FILE__, __LINE__, __func__);
		delete event;
	}
};
inline std::unique_ptr<cudaEvent_t, event_deleter> get_event_unique_ptr(){
	auto event = new cudaEvent_t;
	cutf::error::check(cudaEventCreate(event), __FILE__, __LINE__, __func__);
	return std::unique_ptr<cudaEvent_t, event_deleter>{event};
}

float get_elapsed_time(const cudaEvent_t start_event, const cudaEvent_t end_event) {
	float elapsed_time;
	cutf::error::check(cudaEventElapsedTime(&elapsed_time, start_event, end_event), __FILE__, __LINE__, __func__);
	return elapsed_time;
}
} // namespace event
} // namespace cutf

#endif /* end of include guard */
