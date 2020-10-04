#include <cutf/threads.hpp>
#include <iostream>

__global__ void get_lane_id_kernel() {
	const unsigned thread_id = threadIdx.x;
	const unsigned lane_id = cutf::threads::get_lane_id();
	const unsigned warp_id = cutf::threads::get_warp_id();
	std::printf("threadIdx.x = %u, lane_id = %u, warp_id = %u\n", thread_id, lane_id, warp_id);
}

int main(){
	get_lane_id_kernel<<<1, 64>>>();
	cudaDeviceSynchronize();
}
