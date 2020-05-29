#ifndef __CUTF_EVENT_HPP__
#define __CUTF_EVENT_HPP__
#include "cuda.hpp"

namespace cutf {
namespace event {
struct event_deleter{
	void operator()(cudaEvent_t* event){
		cutf::error::check(cudaEventDestroy(*event), __FILE__, __LINE__, __func__);
		delete handle;
	}
};
inline std::unique_ptr<cudaEvent_t, event_deleter> get_event_unique_ptr(){
	cudaEvent_t* event = new cudaEvent_t;
	cutf::error::check(cudaEventCreate(event), __FILE__, __LINE__, __func__);
	return std::unique_ptr<cudaEvent_t, event_deleter>{event};
}
} // namespace event
} // namespace cutf

#endif /* end of include guard */
