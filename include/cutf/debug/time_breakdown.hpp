#ifndef __CUTF_TIME_BREAKDOWN_HPP__
#define __CUTF_TIME_BREAKDOWN_HPP__
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "../cuda.hpp"
#include "../error.hpp"

namespace cutf {
namespace debug {
namespace time_breakdown {
class profiler {
	std::unordered_map<std::string, std::chrono::system_clock::time_point> start_timestamps;
	std::unordered_map<std::string, std::vector<std::time_t>> elapsed_time_list_table;
	cudaStream_t cuda_stream;

	bool enabled;
public:
	profiler(cudaStream_t cuda_stream = 0) :
		cuda_stream(cuda_stream), enabled(true) {}

	void disable_measurement() {enabled = false;}
	void enable_measurement() {enabled = true;}
	bool is_measure_enabled() const {return enabled;}

	void start_timer_sync(
			const std::string name			
			) {
		if (enabled) {
			CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

			const auto timestamp = std::chrono::system_clock::now();
			if (start_timestamps.count(name) == 0) {
				start_timestamps.insert(std::make_pair(name, timestamp));
			} else {
				start_timestamps[name] = timestamp;
			}
		}
	}

	void set_cuda_stream(cudaStream_t stream) {cuda_stream = stream;}

	void clear() {
		start_timestamps.clear();
		elapsed_time_list_table.clear();
	}

	void stop_timer_sync(
			const std::string name			
			) {
		if (enabled) {
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
	}

	struct statistic_t {
		std::string name;
		std::size_t n;
		std::time_t sum;
		std::time_t min;
		std::time_t max;
		std::time_t median;
	};

	std::vector<statistic_t> get_statistics_list() const {
		std::vector<statistic_t> statistic_list;
		for (auto t : elapsed_time_list_table) {
			statistic_t s{t.first, t.second.size(), 0, 0x7fffffffffffffffl, 0};
			std::sort(t.second.begin(), t.second.end());
			for (const auto &ti : t.second) {
				s.sum += ti;
			}
			s.min = t.second[0];
			s.max = t.second[t.second.size() - 1];
			if ((t.second.size() % 2) == 1) {
				s.median = t.second[t.second.size() / 2];
			} else {
				s.median = (t.second[t.second.size() / 2 - 1] + t.second[t.second.size() / 2]) / 2;
			}
			statistic_list.push_back(s);
		}
		return statistic_list;
	}

	void print_result(FILE* const out = stderr) const {
		// Find the longest entry name
		std::size_t longest_name_length = 0;
		for (const auto& t : elapsed_time_list_table) {
			longest_name_length = std::max(t.first.length(), longest_name_length);
		}

		auto statistic_list = get_statistics_list();
		std::time_t time_total = 0;
		for (const auto s : statistic_list) {
			time_total += s.sum;
		}

		std::sort(statistic_list.begin(), statistic_list.end(),
				[&](const statistic_t& a, const statistic_t& b) {return a.sum > b.sum;}
				);

		std::printf("# cutf time breakdown result (Total: %10.3f [ms])\n", time_total * 1e-6);
		std::printf("%*s %13s           %10s %10s %10s %10s %10s\n",
				static_cast<int>(longest_name_length), "Name",
				"Total [ms]",
				"N",
				"Avg [ms]",
				"Min [ms]",
				"Max [ms]",
				"Med [ms]"
				);
		for (const auto& s : statistic_list) {
			std::printf("%*s %13.3f (%6.2f%%) %10lu %10.3f %10.3f %10.3f %10.3f\n",
					static_cast<int>(longest_name_length), s.name.c_str(),
					s.sum * 1e-6,
					s.sum * 100.0 / time_total,
					s.n,
					s.sum * 1e-6 / s.n,
					s.min * 1e-6,
					s.max * 1e-6,
					s.median * 1e-6
					);
		}
	}

	void print_result_csv(FILE* const out = stderr) const {
		// Find the longest entry name
		struct statistic_t {
			std::string name;
			std::size_t n;
			std::time_t sum;
			std::time_t min;
			std::time_t max;
			std::time_t median;
		};

		auto statistic_list = get_statistics_list();

		std::printf("name,n,sum_us,avg_us,min_us,max_us,med_us\n");
		for (const auto& s : statistic_list) {
			std::printf("%s,%lu,%.3f,%.3f,%.3f,%.3f,%.3f\n",
					s.name.c_str(),
					s.n,
					s.sum * 1e-3 ,
					s.sum * 1e-3  / s.n,
					s.min * 1e-3 ,
					s.max * 1e-3,
					s.median * 1e-3
					);
		}
	}

	void measure(
			const std::string name,
			const std::function<void(void)> func
			) {
		start_timer_sync(name);
		func();
		stop_timer_sync(name);
	}
};
} // namespace time_breakdown
} // namespace debug
} // namespace cutf
#endif
