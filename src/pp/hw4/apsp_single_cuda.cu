#include "apsp.hpp"

namespace pp {
namespace hw4 {

__device__ void CalcBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start, unsigned block_x_len, unsigned block_y_len) {
	// Plan: We can map 1 APSP block to 1 CUDA block.
	// A value of a block is assigned to a CUDA thread of a CUDA block.
	// It needs to be looped k times for k middle nodes.
	// Each loop should have a synchronized barrier in the end.

	// Make sure it is inside the range
	if (blockIdx.x >= block_x_len || blockIdx.y >= block_y_len)
		return;

	// Calculate the block index
	unsigned bx = block_x_start + blockIdx.x;
	unsigned by = block_y_start + blockIdx.y;

	// Calculate the node index
	unsigned src_idx = bx * block_size + threadIdx.x;
	unsigned dst_idx = by * block_size + threadIdx.y;

	// Make sure it is inside the range
	if (src_idx >= num_nodes || dst_idx >= num_nodes)
		return;

	// Calcuate the start and the end index of middle nodes
	unsigned mid_start_idx = block_size * round_idx;
	unsigned mid_end_idx = ((block_size * (round_idx + 1)) < num_nodes)? (block_size * (round_idx + 1)) : num_nodes;

	Cost newCost;
	for (unsigned mid_idx = mid_start_idx + 0; mid_idx < mid_end_idx; mid_idx++) {
		newCost = costs[src_idx * num_nodes + mid_idx] + costs[mid_idx * num_nodes + dst_idx];
		if (newCost < costs[src_idx * num_nodes + dst_idx])
			costs[src_idx * num_nodes + dst_idx] = newCost;

		// Synchronized
		__syncthreads();
	}
}

__global__ void BlockedAPSP(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned num_rounds) {
	// TODO: (Opt.) Declare the variables in shared memory

	// TODO: (Opt.) Copy the data from Global to Shared Memory

	// Multiple rounds
	for (unsigned round_idx = 0; round_idx < num_rounds; round_idx++) {
		unsigned rp1 = round_idx + 1;
		unsigned rr1 = num_rounds - round_idx - 1;

		// Phase 1
		CalcBlocks(costs, num_nodes, block_size, round_idx, round_idx, round_idx, 1, 1);
		__syncthreads();

		// Phase 2
		// Up
		CalcBlocks(costs, num_nodes, block_size, round_idx, round_idx, 0, 1, round_idx);
		// Left
		CalcBlocks(costs, num_nodes, block_size, round_idx, 0, round_idx, round_idx, 1);
		// Right
		CalcBlocks(costs, num_nodes, block_size, round_idx, rp1, round_idx, rr1, 1);
		// Down
		CalcBlocks(costs, num_nodes, block_size, round_idx, round_idx, rp1, 1, rr1);
		__syncthreads();

		// Phase 3
		// Left-Up
		CalcBlocks(costs, num_nodes, block_size, round_idx, 0, 0, round_idx, round_idx);
		// Right-Up
		CalcBlocks(costs, num_nodes, block_size, round_idx, rp1, 0, rr1, round_idx);
		// Left-Down
		CalcBlocks(costs, num_nodes, block_size, round_idx, 0, rp1, round_idx, rr1);
		// Right-Down
		CalcBlocks(costs, num_nodes, block_size, round_idx, rp1, rp1, rr1, rr1);
		__syncthreads();
	}
}

void CalcAPSP(Graph *graph, unsigned block_size) {
	unsigned nvertices = graph->num_vertices;

	// Device (GPU) Initialization
	//cudaSetDevice(0);

	// Allocate memory on GPU
	Cost *costs_on_gpu;
	unsigned data_size = sizeof(Cost) * nvertices * nvertices;
	cudaMalloc((void **) &costs_on_gpu, data_size);

	// Copy the graph from Host to Device
	cudaMemcpy(costs_on_gpu, graph->weights, data_size, cudaMemcpyHostToDevice);

	// Call blocked-APSP kernel
	unsigned num_rounds = (nvertices % block_size == 0)? nvertices / block_size : nvertices / block_size + 1;
	dim3 num_blocks(num_rounds, num_rounds);
	dim3 num_threads(block_size, block_size);
	BlockedAPSP<<<num_blocks, num_threads>>>(costs_on_gpu, nvertices, block_size, num_rounds);
	cudaThreadSynchronize();

	// Copy the result from Device to Host
	cudaMemcpy(graph->weights, costs_on_gpu, data_size, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(costs_on_gpu);
}

}
}
