#ifndef __CUTF_TIME_BREAKDOWN_HPP__
#define __CUTF_TIME_BREAKDOWN_HPP__
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <map>
#include "../cuda.hpp"
#include "../error.hpp"

namespace cutf {
namespace debug {
namespace time_breakdown {
class profiler {
	std::map<std::string, std::chrono::system_clock::time_point> start_timestamps;
	std::map<std::string, std::vector<std::time_t>> elapsed_time_list_table;
	cudaStream_t cuda_stream;
public:
	profiler(cudaStream_t cuda_stream = 0) :
		cuda_stream(cuda_stream) {}
	void start_timer_sync(
			const std::string name			
			) {
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

		const auto timestamp = std::chrono::system_clock::now();
		if (start_timestamps.count(name) == 0) {
			start_timestamps.insert(std::make_pair(name, timestamp));
		} else {
			start_timestamps[name] = timestamp;
		}
	}

	void set_cuda_stream(cudaStream_t stream) {cuda_stream = stream;}

	void stop_timer_sync(
			const std::string name			
			) {
		CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

		const auto timestamp = std::chrono::system_clock::now();
		if (start_timestamps.count(name) == 0) {
			throw std::runtime_error("Timer \"" + name + "\" is not started");
		} else {
			const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp - start_timestamps[name]).count();
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
		struct statistic_t {
			std::string name;
			std::size_t n;
			std::time_t sum;
			std::time_t min;
			std::time_t max;
		};

		std::vector<statistic_t> statistic_list;
		for (const auto& t : elapsed_time_list_table) {
			statistic_t s{t.first, t.second.size(), 0, -1, 0};
			for (const auto &ti : t.second) {
				s.sum += ti;
				s.max = std::max(s.max, ti);
				s.min = std::min(s.max, ti);
			}
			statistic_list.push_back(s);
		}

		std::sort(statistic_list.begin(), statistic_list.end(),
				[&](const statistic_t& a, const statistic_t& b) {return a.sum > b.sum;}
				);

			std::printf("%*s  %13s %10s %10s %10s %10s\n",
					static_cast<int>(longest_name_length), "Name",
					"Total [ms]",
					"N",
					"Avg [ms]",
					"Min [ms]",
					"Max [ms]");
		for (const auto& s : statistic_list) {
			std::printf("%*s  %13.3f %10lu %10.3f %10.3f %10.3f\n",
					static_cast<int>(longest_name_length), s.name.c_str(),
					s.sum / 1000.0,
					s.n,
					s.sum / s.n / 1000.0,
					s.min / 1000.0,
					s.max / 1000.0
					);
		}
	}
};
} // namespace time_breakdown
} // namespace debug
} // namespace cutf
#endif
