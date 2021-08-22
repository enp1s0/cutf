#include <cutf/graph.hpp>
#include <cutf/memory.hpp>
#include <cutf/stream.hpp>
#include <vector>

constexpr std::size_t N = 1lu << 20;
constexpr std::size_t block_size = 256;

__global__ void vector_add_kernel(
		float* const c_ptr,
		const float* const a_ptr,
		const float* const b_ptr,
		const std::size_t N
		) {
	const auto tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid >= N) return;

	c_ptr[tid] = a_ptr[tid] + b_ptr[tid];
}

int main() {
	auto da_uptr = cutf::memory::get_device_unique_ptr<float>(N);
	auto db_uptr = cutf::memory::get_device_unique_ptr<float>(N);
	auto dc_uptr = cutf::memory::get_device_unique_ptr<float>(N);
	auto ha_uptr = cutf::memory::get_host_unique_ptr<float>(N);
	auto hb_uptr = cutf::memory::get_host_unique_ptr<float>(N);
	auto hc_uptr = cutf::memory::get_host_unique_ptr<float>(N);

	for (std::size_t i = 0; i < N; i++) {
		ha_uptr.get()[i] = i;
		hb_uptr.get()[i] = i;
		hc_uptr.get()[i] = 0;
	}

	auto cuda_graph = cutf::graph::get_graph_unique_ptr();
	cudaStream_t cuda_stream;
	CUTF_CHECK_ERROR(cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking));
	std::vector<cudaGraphNode_t> node_dependencies;

	cudaMemcpy3DParms memcpy_params = {0};
	cudaKernelNodeParams kernel_node_params = {0};
	cudaGraphNode_t memcpy_node, kernel_node;

	// Copy A
	memcpy_params = cutf::graph::get_simple_memcpy_params(da_uptr.get(), ha_uptr.get(), N, cudaMemcpyHostToDevice);
	CUTF_CHECK_ERROR(cudaGraphAddMemcpyNode(&memcpy_node, *cuda_graph.get(), nullptr, 0, &memcpy_params));
	node_dependencies.push_back(memcpy_node);

	// Copy B
	memcpy_params = cutf::graph::get_simple_memcpy_params(db_uptr.get(), hb_uptr.get(), N, cudaMemcpyHostToDevice);
	CUTF_CHECK_ERROR(cudaGraphAddMemcpyNode(&memcpy_node, *cuda_graph.get(), nullptr, 0, &memcpy_params));
	node_dependencies.push_back(memcpy_node);

	// Kernel
	auto size = N;
	auto da_ptr = da_uptr.get();
	auto db_ptr = db_uptr.get();
	auto dc_ptr = dc_uptr.get();
	void* kernel_args[4] = {reinterpret_cast<void*>(&dc_ptr), reinterpret_cast<void*>(&da_ptr), reinterpret_cast<void*>(&db_ptr), reinterpret_cast<void*>(&size)};
	kernel_node_params = cutf::graph::get_simple_kernel_node_params(
			reinterpret_cast<void*>(vector_add_kernel),
			kernel_args,
			dim3(N / block_size, 1, 1),
			dim3(block_size, 1, 1)
			);
	CUTF_CHECK_ERROR(cudaGraphAddKernelNode(&kernel_node, *cuda_graph.get(), node_dependencies.data(), node_dependencies.size(), &kernel_node_params));
	node_dependencies.push_back(kernel_node);

	// Copy C
	memcpy_params = cutf::graph::get_simple_memcpy_params(hc_uptr.get(), dc_uptr.get(), N, cudaMemcpyDeviceToHost);
	CUTF_CHECK_ERROR(cudaGraphAddMemcpyNode(&memcpy_node, *cuda_graph.get(), node_dependencies.data(), node_dependencies.size(), &memcpy_params));
	node_dependencies.push_back(memcpy_node);

	// Execute
	auto cuda_graph_exec = cutf::graph::get_graph_exec_unique_ptr(*cuda_graph.get());

	CUTF_CHECK_ERROR(cudaGraphLaunch(*cuda_graph_exec.get(), cuda_stream));
	CUTF_CHECK_ERROR(cudaStreamSynchronize(cuda_stream));

	cudaDeviceSynchronize();
	double max_error = 0.;
	for (std::size_t i = 0; i < 100; i++) {
		const double ref = ha_uptr.get()[i] + hb_uptr.get()[i];
		const auto diff = ref - hc_uptr.get()[i];
		max_error = std::max(max_error, std::abs(diff));
	}
	std::printf("[cudaGraph] error = %e\n", max_error);
}
