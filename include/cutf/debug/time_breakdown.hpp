#ifndef __CUTF_TIME_BREAKDOWN_HPP__
#define __CUTF_TIME_BREAKDOWN_HPP__
#include <vector>
#include <chrono>
#include <string>
#include <map>
#include "../cuda.hpp"
#include "../error.hpp"

namespace cutf {
namespace debug {
namespace time_breakdown {
class profiler {
	std::map<std::string, std::chrono::system_clock::time_point> start_timestamps;
	std::map<std::string, std::vector<std::time_t>> elapsed_time_list_table;
public:
	void register_start_timer_sync_device(
			const std::string name			
			) {
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());

		const auto timestamp = std::chrono::system_clock::now();
		if (start_timestamps.count(name) == 0) {
			start_timestamps.insert(std::make_pair(name, timestamp));
		} else {
			start_timestamps[name] = timestamp;
		}
	}

	void register_stop_timer_sync_device(
			const std::string name			
			) {
		CUTF_CHECK_ERROR(cudaDeviceSynchronize());

		const auto timestamp = std::chrono::system_clock::now();
		if (start_timestamps.count(name) == 0) {
			throw std::runtime_error("Timer \"" + name + "\" is not started");
		} else {
			const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(start_timestamps[name] - timestamp).count();
			if (elapsed_time_list_table.count(name) == 0) {
				std::vector<std::time_t> tmp_elapsed_time_list = {elapsed_time};
				elapsed_time_list_table.insert(std::make_pair(name, tmp_elapsed_time_list));
			} else {
				elapsed_time_list_table[name].push_back(elapsed_time);
			}
			start_timestamps.erase(name);
		}
	}

	void print_result(FILE* const out = stderr) const {
		// Find the longest entry name
		std::size_t longest_name_length = 0;
		for (const auto& t : elapsed_time_list_table) {
			longest_name_length = std::max(t.first.length(), longest_name_length);
		}
	}
};
} // namespace time_breakdown
} // namespace debug
} // namespace cutf
#endif
