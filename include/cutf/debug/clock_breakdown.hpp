#ifndef __CUTF_DEBUG_CLOCK_BREAKDOWN_HPP__
#define __CUTF_DEBUG_CLOCK_BREAKDOWN_HPP__

#define CUTF_CLOCK_BREAKDOWN_INIT(num_timestamps) \
	long long int _cutf_timestamp_list[num_timestamps]

#define CUTF_CLOCK_BREAKDOWN_RECORD(timestamp_index) \
	_cutf_timestamp_list[timestamp_index] = clock64()

#define CUTF_CLOCK_BREAKDOWN_DURATION(start_index, end_index) \
	(_cutf_timestamp_list[end_index] - _cutf_timestamp_list[start_index])

#endif
