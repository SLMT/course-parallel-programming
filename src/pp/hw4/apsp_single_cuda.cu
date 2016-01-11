#include "apsp.hpp"

#include "io.hpp"

namespace pp {
namespace hw4 {

// Pinned Memory Optimization
Cost *NewCosts(unsigned num_costs) {
	Cost *costs;
	cudaMallocHost((void **) &costs, num_costs * sizeof(Cost));
	return costs;
}

void DeleteCosts(Cost *costs) {
	cudaFreeHost(costs);
}

// Shared Memory Optimization
extern __shared__ Cost costs_in_sm[];

__device__ void CalcABlock(Cost *self, Cost *depen1, Cost *depen2, unsigned block_size, unsigned num_mid) {
	// Plan: We can map 1 APSP block to 1 CUDA block.
	// A value of a block is assigned to a CUDA thread of a CUDA block.
	// It needs to be looped k times for k middle nodes.
	// Each loop should have a synchronized barrier in the end.

	Cost cost1, cost2, final_cost;
	unsigned src = threadIdx.x;
	unsigned dst = threadIdx.y;

	for (unsigned mid = 0; mid < num_mid; mid++) {
		// Find the smaller cost
		cost1 = self[src * block_size + dst];
		cost2 = depen1[src * block_size + mid] + depen2[mid * block_size + dst];
		final_cost = (cost1 < cost2)? cost1 : cost2;

		// Synchronized
		__syncthreads();

		// Save the new cost back to shared memory
		self[src * block_size + dst] = final_cost;

		// Synchronized
		__syncthreads();
	}
}

__device__ void CopyCostFromGlobalToSM(Cost *gl, Cost *sm, unsigned num_nodes, unsigned block_size, unsigned bx, unsigned by) {
	unsigned gx = bx * block_size + threadIdx.x;
	unsigned gy = by * block_size + threadIdx.y;
	unsigned sx = threadIdx.x;
	unsigned sy = threadIdx.y;

	if (gx < num_nodes && gy < num_nodes)
		sm[sx * block_size + sy] = gl[gx * num_nodes + gy];
}

__device__ void CopyCostFromSMToGlobal(Cost *gl, Cost *sm, unsigned num_nodes, unsigned block_size, unsigned bx, unsigned by) {
	unsigned gx = bx * block_size + threadIdx.x;
	unsigned gy = by * block_size + threadIdx.y;
	unsigned sx = threadIdx.x;
	unsigned sy = threadIdx.y;

	if (gx < num_nodes && gy < num_nodes)
		gl[gx * num_nodes + gy] = sm[sx * block_size + sy];
}

__global__ void CalcBlocksOnGPU(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start) {
	unsigned gx = (block_x_start + blockIdx.x) * block_size + threadIdx.x;
	unsigned gy = (block_y_start + blockIdx.y) * block_size + threadIdx.y;
	unsigned num_mid = ((round_idx + 1) * block_size < num_nodes)? block_size : num_nodes - round_idx * block_size;

	// Copy self data to Shared Memory
	Cost *self_in_sm = costs_in_sm;
	CopyCostFromGlobalToSM(costs, self_in_sm, num_nodes, block_size, block_x_start + blockIdx.x, block_y_start + blockIdx.y);

	// Copy the first dependent data to Shared Memory
	Cost *depen1_in_sm;
	if (round_idx == block_y_start + blockIdx.y) {
		depen1_in_sm = self_in_sm;
	} else {
		depen1_in_sm = costs_in_sm + block_size * block_size;
		CopyCostFromGlobalToSM(costs, depen1_in_sm, num_nodes, block_size, block_x_start + blockIdx.x, round_idx);
	}

	// Copy the second dependent data to Shared Memory
	Cost *depen2_in_sm;
	if (round_idx == block_x_start + blockIdx.x) {
		depen2_in_sm = self_in_sm;
	} else {
		depen2_in_sm = costs_in_sm + 2 * block_size * block_size;
		CopyCostFromGlobalToSM(costs, depen2_in_sm, num_nodes, block_size, round_idx, block_y_start + blockIdx.y);
	}

	// Synchronized
	// Important!!! Wait for completion of copying data
	// Otherwise, the computation will start before copying finishs
	__syncthreads();

	// Calculate the block
	if (gx < num_nodes && gy < num_nodes)
		CalcABlock(self_in_sm, depen1_in_sm, depen2_in_sm, block_size, num_mid);

	// Synchronized
	__syncthreads();

	// Move the self data back to Global
	CopyCostFromSMToGlobal(costs, self_in_sm, num_nodes, block_size, block_x_start + blockIdx.x, block_y_start + blockIdx.y);
}

void CalcBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start, unsigned block_x_len, unsigned block_y_len) {
	dim3 num_blocks(block_x_len, block_y_len);
	dim3 num_threads(block_size, block_size);
	unsigned sm_size = 3 * block_size * block_size * sizeof(Cost);

	CalcBlocksOnGPU<<<num_blocks, num_threads, sm_size>>>(costs, num_nodes, block_size, round_idx, block_x_start, block_y_start);
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

	// Blocked-APSP Algorithm
	unsigned num_rounds = (nvertices % block_size == 0)? nvertices / block_size : nvertices / block_size + 1;
	for (unsigned round_idx = 0; round_idx < num_rounds; round_idx++) {
		unsigned rp1 = round_idx + 1;
		unsigned rr1 = num_rounds - round_idx - 1;

		// Phase 1
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, round_idx, round_idx, 1, 1);
		// Wait for complete
		cudaThreadSynchronize();

		// Phase 2
		// Up
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, round_idx, 0, 1, round_idx);
		// Left
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, 0, round_idx, round_idx, 1);
		// Right
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, rp1, round_idx, rr1, 1);
		// Down
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, round_idx, rp1, 1, rr1);
		// Wait for complete
		cudaThreadSynchronize();

		// Phase 3
		// Left-Up
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, 0, 0, round_idx, round_idx);
		// Right-Up
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, rp1, 0, rr1, round_idx);
		// Left-Down
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, 0, rp1, round_idx, rr1);
		// Right-Down
		CalcBlocks(costs_on_gpu, nvertices, block_size, round_idx, rp1, rp1, rr1, rr1);
		// Wait for complete
		cudaThreadSynchronize();
	}

	// Copy the result from Device to Host
	cudaMemcpy(graph->weights, costs_on_gpu, data_size, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(costs_on_gpu);
}

}
}
