#include "apsp.hpp"

#include "block_calculation.hpp"

namespace pp {
namespace hw4 {

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

} // namespace hw4
} // namespace pp
