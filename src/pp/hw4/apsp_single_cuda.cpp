#include "apsp.hpp"

namespace pp {
namespace hw4 {

__device__ void CalcBlocks(Cost *costs, unsigned block_size, unsigned round_num, unsigned block_x_start, unsigned block_y_start, unsigned block_x_len, unsigned block_y_len) {
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
	unsigned src_index = bx * block_size + threadIdx.x;
	unsigned dest_index = by * block_size + threadIdx.y;

	Cost newCost;
	for (unsigned mid_index/* TODO: k times */) {
		newCost = costs[src_index][mid_index] + costs[mid_index][dest_index];
		if (newCost < costs[src_index][dest_index])
			costs[src_index][dest_index] = newCost;

		// TODO: Synchronized
	}
}

__global__ void BlockedAPSP(Cost *costs, unsigned num_node, unsigned block_size) {
	// TODO: (Opt.) Declare the variables in shared memory

	// TODO: (Opt.) Copy the data from Global to Shared Memory

	// TODO: Multiple rounds
		// TODO: Phase 1

		// TODO: Synchronized

		// TODO: Phase 2

		// TODO: Synchronized

		// TODO: Phase 3

		// TODO: Synchronized
}

void CalcAPSP(Graph *graph, unsigned block_size) {
	unsigned nvertices = graph->num_vertices;
	Cost *weights = graph->weights;

	Cost new_cost;
	for (unsigned k = 0; k < nvertices; k++) {
		for (unsigned i = 0; i < nvertices; i++) {
			for (unsigned j = 0; j < nvertices; j++) {
				new_cost = weights[i * nvertices + k] + weights[k * nvertices + j];

				if (weights[i * nvertices + j] > new_cost) {
					weights[i * nvertices + j] = new_cost;
				}
			}
		}
	}

	// Allocate memory on GPU
	Cost *costs_on_gpu;
	sizt_t data_size = nvertices * nvertices;
	cudaMalloc((void **) &costs_on_gpu, data_size);

	// Copy the graph from Host to Device
	cudaMemcpy(costs_on_gpu, graph->weights, data_size, cudaMemcpyHostToDevice);

	// TODO: Call blocked-APSP kernel
	size_t num_blocks =
	size_t num_thread =
	BlockedAPSP<<<num_blocks, num_thread>>>();
	cudaThreadSynchronize();

	// Copy the result from Device to Host
	cudaMemcpy(graph->weights, costs_on_gpu, data_size, cudaMemcpyDeviceToHost);

	// Free memory on GPU
	cudaFree(costs_on_gpu);
}

// =====================================================
// =============== Example Code From TAs ===============
// =====================================================

int ceil(int a, int b)
{
	return (a + b -1)/b;
}

void block_FW(int B)
{
	int round = ceil(n, B);
	for (int r = 0; r < round; ++r) {
		/* Phase 1*/
		cal(B,	r,	r,	r,	1,	1);

		/* Phase 2*/
		cal(B, r,     r,     0,             r,             1);
		cal(B, r,     r,  r +1,  round - r -1,             1);
		cal(B, r,     0,     r,             1,             r);
		cal(B, r,  r +1,     r,             1,  round - r -1);

		/* Phase 3*/
		cal(B, r,     0,     0,            r,             r);
		cal(B, r,     0,  r +1,  round -r -1,             r);
		cal(B, r,  r +1,     0,            r,  round - r -1);
		cal(B, r,  r +1,  r +1,  round -r -1,  round - r -1);
	}
}

void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height)
{
	int block_end_x = block_start_x + block_height;
	int block_end_y = block_start_y + block_width;

	for (int b_i =  block_start_x; b_i < block_end_x; ++b_i) {
		for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
			// To calculate B*B elements in the block (b_i, b_j)
			// For each block, it need to compute B times
			for (int k = Round * B; k < (Round +1) * B && k < n; ++k) {
				// To calculate original index of elements in the block (b_i, b_j)
				// For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
				int block_internal_start_x = b_i * B;
				int block_internal_end_x   = (b_i +1) * B;
				int block_internal_start_y = b_j * B;
				int block_internal_end_y   = (b_j +1) * B;

				if (block_internal_end_x > n)	block_internal_end_x = n;
				if (block_internal_end_y > n)	block_internal_end_y = n;

				for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
					for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
						if (Dist[i][k] + Dist[k][j] < Dist[i][j])
							Dist[i][j] = Dist[i][k] + Dist[k][j];
					}
				}
			}
		}
	}
}

}
}
