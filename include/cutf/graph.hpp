#ifndef __CUTF_GRAPH_HPP__
#define __CUTF_GRAPH_HPP__
#include <memory>
#include "cuda.hpp"

namespace cutf {
namespace graph {
struct graph_deleter{
	void operator()(cudaGraph_t* graph){
		cutf::error::check(cudaGraphDestroy(*graph), __FILE__, __LINE__, __func__);
		delete graph;
	}
};
inline std::unique_ptr<cudaGraph_t, graph_deleter> get_graph_unique_ptr(){
	auto graph = new cudaGraph_t;
	cutf::error::check(cudaGraphCreate(graph, 0), __FILE__, __LINE__, __func__);
	return std::unique_ptr<cudaGraph_t, graph_deleter>{graph};
}

struct graph_exec_deleter {
	void operator()(cudaGraphExec_t* exec){
		cutf::error::check(cudaGraphExecDestroy(*exec), __FILE__, __LINE__, __func__);
		delete exec;
	}
};
inline std::unique_ptr<cudaGraphExec_t, graph_exec_deleter> get_graph_exec_unique_ptr(cudaGraph_t graph){
	auto graph_exec = new cudaGraphExec_t;
	cutf::error::check(cudaGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0), __FILE__, __LINE__, __func__);
	return std::unique_ptr<cudaGraphExec_t, graph_exec_deleter>{graph_exec};
}

template <class T>
inline cudaMemcpy3DParms get_simple_memcpy_params(
		T* const dst_ptr,
		T* const src_ptr,
		const std::size_t num_elements,
		const  cudaMemcpyKind kind
		) {
	cudaMemcpy3DParms memcpy_params;
	memcpy_params.srcArray = nullptr;
	memcpy_params.srcPos   = make_cudaPos(0, 0, 0);
	memcpy_params.srcPtr   = make_cudaPitchedPtr(reinterpret_cast<void*>(src_ptr), sizeof(float) * num_elements, num_elements, 1);
	memcpy_params.dstArray = nullptr;
	memcpy_params.dstPos   = make_cudaPos(0, 0, 0);
	memcpy_params.dstPtr   = make_cudaPitchedPtr(reinterpret_cast<void*>(dst_ptr), sizeof(float) * num_elements, num_elements, 1);
	memcpy_params.extent   = make_cudaExtent(num_elements * sizeof(T), 1, 1);
	memcpy_params.kind     = kind;

	return memcpy_params;
}

inline cudaKernelNodeParams get_simple_kernel_node_params(
		void* kernel_ptr,
		void** kernel_args,
		const dim3 grid_size,
		const dim3 block_size,
		const unsigned shmem_size = 0
		) {
	cudaKernelNodeParams kernel_params;
	kernel_params.func           = kernel_ptr;
	kernel_params.gridDim        = grid_size;
	kernel_params.blockDim       = block_size;
	kernel_params.sharedMemBytes = shmem_size;
	kernel_params.kernelParams   = kernel_args;
	kernel_params.extra          = nullptr;

	return kernel_params;
}
} // namespace graph
} // namespce cutf
#endif
