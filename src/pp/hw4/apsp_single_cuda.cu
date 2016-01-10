#include "apsp.hpp"

namespace pp {
namespace hw4 {

__device__ void CalcABlock(Cost *self, Cost *depen1, Cost *depen2, unsigned block_size, unsigned num_mid) {
	// Plan: We can map 1 APSP block to 1 CUDA block.
	// A value of a block is assigned to a CUDA thread of a CUDA block.
	// It needs to be looped k times for k middle nodes.
	// Each loop should have a synchronized barrier in the end.

	Cost cost1, cost2, final_cost;
	unsigned src_idx = threadIdx.x;
	unsigned dst_idx = threadIdx.y;

	for (unsigned mid_idx = 0; mid_idx < num_mid; mid_idx++) {
		// Find the smaller cost
		cost1 = self[src_idx * block_size + dst_idx];
		cost2 = depen1[src_idx * block_size + mid_idx] + depen2[mid_idx * block_size + dst_idx];
		final_cost = (cost1 < cost2)? cost1 : cost2;

		// Synchronized
		__syncthreads();

		// Save the new cost back to shared memory
		self[src_idx * block_size + dst_idx] = final_cost;
	}
}

// A shared memory variable
extern __shared__ Cost costs_in_sm[];

__device__ void CopyCostFromGlobalToSM(Cost *gl, Cost *sm, unsigned block_size, unsigned bx, unsigned by) {
	unsigned gx = bx * block_size + threadIdx.x;
	unsigned gy = by * block_size + threadIdx.y;
	unsigned sx = threadIdx.x;
	unsigned sy = threadIdx.y;

	sm[sx * block_size + sy] = gl[gx * block_size + gy];
}

__device__ void CopyCostFromSMToGlobal(Cost *gl, Cost *sm, unsigned block_size, unsigned bx, unsigned by) {
	unsigned gx = bx * block_size + threadIdx.x;
	unsigned gy = by * block_size + threadIdx.y;
	unsigned sx = threadIdx.x;
	unsigned sy = threadIdx.y;

	gl[gx * block_size + gy] = sm[sx * block_size + sy];
}

__global__ void CalcIndependBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx) {
	unsigned gx = round_idx * block_size + threadIdx.x;
	unsigned gy = round_idx * block_size + threadIdx.y;
	unsigned num_mid = ((round_idx + 1) * block_size < num_nodes)? block_size : num_nodes - round_idx * block_size;

	if (gx < num_nodes && gy < num_nodes) {
		// Move the data from Global to Shared Memory
		CopyCostFromGlobalToSM(costs, costs_in_sm, round_idx, round_idx, block_size);

		// Calculate the block
		CalcABlock(costs_in_sm, costs_in_sm, costs_in_sm, block_size, num_mid);

		// Move the data back to Global
		CopyCostFromSMToGlobal(costs, costs_in_sm, round_idx, round_idx, block_size);
	}
}

__global__ void CalcSinglyDependBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start) {
	unsigned gx = (block_x_start + blockIdx.x) * block_size + threadIdx.x;
	unsigned gy = (block_y_start + blockIdx.y) * block_size + threadIdx.y;
	unsigned num_mid = ((round_idx + 1) * block_size < num_nodes)? block_size : num_nodes - round_idx * block_size;
	Cost *self_in_sm = costs_in_sm;
	Cost *depen_in_sm = costs_in_sm + block_size * block_size;

	if (gx < num_nodes && gy < num_nodes) {
		// Move the data from Global to Shared Memory
		CopyCostFromGlobalToSM(costs, self_in_sm, block_x_start + blockIdx.x, block_y_start + blockIdx.y, block_size);
		CopyCostFromGlobalToSM(costs, depen_in_sm, round_idx, round_idx, block_size);

		// Calculate the block
		CalcABlock(self_in_sm, depen_in_sm, self_in_sm, block_size, num_mid);

		// Move the data back to Global
		CopyCostFromSMToGlobal(costs, self_in_sm, block_x_start + blockIdx.x, block_y_start + blockIdx.y, block_size);
		CopyCostFromSMToGlobal(costs, depen_in_sm, round_idx, round_idx, block_size);
	}
}

__global__ void CalcDoublyDependBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start) {
	unsigned gx = (block_x_start + blockIdx.x) * block_size + threadIdx.x;
	unsigned gy = (block_y_start + blockIdx.y) * block_size + threadIdx.y;
	unsigned num_mid = ((round_idx + 1) * block_size < num_nodes)? block_size : num_nodes - round_idx * block_size;
	Cost *self_in_sm = costs_in_sm;
	Cost *depen1_in_sm = costs_in_sm + block_size * block_size;
	Cost *depen2_in_sm = costs_in_sm + 2 * block_size * block_size;

	if (gx < num_nodes && gy < num_nodes) {
		// Move the data from Global to Shared Memory
		CopyCostFromGlobalToSM(costs, self_in_sm, block_x_start + blockIdx.x, block_y_start + blockIdx.y, block_size);
		CopyCostFromGlobalToSM(costs, depen1_in_sm, block_x_start + blockIdx.x, round_idx, block_size);
		CopyCostFromGlobalToSM(costs, depen2_in_sm, round_idx, block_y_start + blockIdx.y, block_size);

		// Calculate the block
		CalcABlock(self_in_sm, depen1_in_sm, depen2_in_sm, block_size, num_mid);

		// Move the data back to Global
		CopyCostFromSMToGlobal(costs, self_in_sm, block_x_start + blockIdx.x, block_y_start + blockIdx.y, block_size);
		CopyCostFromSMToGlobal(costs, depen1_in_sm, block_x_start + blockIdx.x, round_idx, block_size);
		CopyCostFromSMToGlobal(costs, depen2_in_sm, round_idx, block_y_start + blockIdx.y, block_size);
	}
}

void CUDACalcIndependBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx) {
	dim3 num_blocks(1, 1);
	dim3 num_threads(block_size, block_size);
	unsigned sm_size = block_size * block_size * sizeof(Cost);

	CalcIndependBlocks<<<num_blocks, num_threads, sm_size>>>(costs, num_nodes, block_size, round_idx);
}

void CUDACalcSinglyDependBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start, unsigned block_x_len, unsigned block_y_len) {
	dim3 num_blocks(block_x_len, block_y_len);
	dim3 num_threads(block_size, block_size);
	unsigned sm_size = 2 * block_size * block_size * sizeof(Cost);

	CalcSinglyDependBlocks<<<num_blocks, num_threads, sm_size>>>(costs, num_nodes, block_size, round_idx, block_x_start, block_y_start);
}

void CUDACalcDoublyDependBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start, unsigned block_x_len, unsigned block_y_len) {
	dim3 num_blocks(block_x_len, block_y_len);
	dim3 num_threads(block_size, block_size);
	unsigned sm_size = 3 * block_size * block_size * sizeof(Cost);

	CalcDoublyDependBlocks<<<num_blocks, num_threads, sm_size>>>(costs, num_nodes, block_size, round_idx, block_x_start, block_y_start);
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
		CUDACalcIndependBlocks(costs_on_gpu, nvertices, block_size, round_idx);
		// Wait for complete
		cudaThreadSynchronize();

		// Phase 2
		// Up
		CUDACalcSinglyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, round_idx, 0, 1, round_idx);
		// Left
		CUDACalcSinglyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, 0, round_idx, round_idx, 1);
		// Right
		CUDACalcSinglyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, rp1, round_idx, rr1, 1);
		// Down
		CUDACalcSinglyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, round_idx, rp1, 1, rr1);
		// Wait for complete
		cudaThreadSynchronize();

		// Phase 3
		// Left-Up
		CUDACalcDoublyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, 0, 0, round_idx, round_idx);
		// Right-Up
		CUDACalcDoublyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, rp1, 0, rr1, round_idx);
		// Left-Down
		CUDACalcDoublyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, 0, rp1, round_idx, rr1);
		// Right-Down
		CUDACalcDoublyDependBlocks(costs_on_gpu, nvertices, block_size, round_idx, rp1, rp1, rr1, rr1);
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
