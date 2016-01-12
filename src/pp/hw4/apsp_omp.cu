#include "apsp.hpp"

#include <cstdio>
#include <omp.h>

#include "block_calculation.hpp"
#include "../timer.hpp"

namespace pp {
namespace hw4 {

typedef struct {
	unsigned x_start, y_start, x_len, y_len;
} Range;

typedef struct {
	Time mem_time, sync_time;
} Profile;

// XXX: This method only works in the case of 2 partition.
// This is not scalable.
void DetermineCalculateRange(Range *range, unsigned x_start, unsigned y_start, unsigned x_len, unsigned y_len) {
	// X length is even
	if (x_len % 2 == 0) {
		Range r;
		r.x_len = x_len / 2;
		r.y_len = y_len;

		range[0] = r;
		range[0].x_start = x_start;
		range[0].y_start = y_start;

		range[1] = r;
		range[1].x_start = x_start + x_len / 2;
		range[1].y_start = y_start;

	// Y length is even (and x_len isn't)
	} else if (y_len % 2 == 0) {
		Range r;
		r.x_len = x_len;
		r.y_len = y_len / 2;

		range[0] = r;
		range[0].x_start = x_start;
		range[0].y_start = y_start;

		range[1] = r;
		range[1].x_start = x_start;
		range[1].y_start = y_start + y_len / 2;

	// X & Y length are both not even.
	} else {
		// Choose the larger side for partition
		if (x_len >= y_len) {
			range[0].x_start = x_start;
			range[0].y_start = y_start;
			range[0].x_len = x_len / 2;
			range[0].y_len = y_len;

			range[1].x_start = x_start + x_len / 2;
			range[1].y_start = y_start;
			range[1].x_len = x_len / 2 + 1;
			range[1].y_len = y_len;
		} else {
			range[0].x_start = x_start;
			range[0].y_start = y_start;
			range[0].x_len = x_len;
			range[0].y_len = y_len / 2;

			range[1].x_start = x_start;
			range[1].y_start = y_start + y_len / 2;
			range[1].x_len = x_len;
			range[1].y_len = y_len / 2 + 1;
		}
	}
}

void CopyAllData(Cost *src, Cost *dst, unsigned num_nodes, unsigned block_size) {
	unsigned num_blocks = (num_nodes % block_size == 0)? num_nodes / block_size : num_nodes / block_size + 1;

	for (unsigned bx = 0; bx < num_blocks; bx++) {
		for (unsigned by = 0; by < num_blocks; by++) {
			for (unsigned i = bx * block_size; i < (bx + 1) * block_size && i < num_nodes; i++) {
				for (unsigned j = by * block_size; j < (by + 1) * block_size && j < num_nodes; j++) {
					dst[i * num_nodes + j] = src[i * num_nodes + j];
				}
			}
		}
	}
}

void CopyData(Cost *src, Cost *dst, Range r, unsigned num_nodes, unsigned block_size) {
	for (unsigned bx = r.x_start; bx < r.x_start + r.x_len; bx++) {
		for (unsigned by = r.y_start; by < r.y_start + r.y_len; by++) {
			for (unsigned i = bx * block_size; i < (bx + 1) * block_size && i < num_nodes; i++) {
				for (unsigned j = by * block_size; j < (by + 1) * block_size && j < num_nodes; j++) {
					dst[i * num_nodes + j] = src[i * num_nodes + j];
				}
			}
		}
	}
}

void ParallelCalcBlocks(Cost *local_buf, Cost *global_buf, Cost *gpu_data, Range *range, unsigned cc_num, unsigned self_id, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned bx, unsigned by, unsigned bxlen, unsigned bylen, Profile *prof) {
	Time start;

	if (bxlen != 0 && bylen != 0) {
		// Determine a range for calculation
		DetermineCalculateRange(range, bx, by, bxlen, bylen);
		Range my_range = range[self_id];

		// Copy the data from Host to Device
		start = GetCurrentTime();
		unsigned data_size = sizeof(Cost) * num_nodes * num_nodes;
		cudaMemcpy(gpu_data, local_buf, data_size, cudaMemcpyHostToDevice);
		prof->mem_time = TimeAdd(prof->mem_time, TimeDiff(start, GetCurrentTime()));

		//printf("Process %d in round %u range (%u, %u, %u, %u)\n", self_id, round_idx, my_range.x_start, my_range.y_start, my_range.x_len, my_range.y_len);

		// Calculate the blocks in the range
		CalcBlocks(gpu_data, num_nodes, block_size, round_idx, my_range.x_start, my_range.y_start, my_range.x_len, my_range.y_len);
		// Wait for complete
		cudaThreadSynchronize();

		// Copy the data from Device to Host
		start = GetCurrentTime();
		cudaMemcpy(local_buf, gpu_data, data_size, cudaMemcpyDeviceToHost);
		prof->mem_time = TimeAdd(prof->mem_time, TimeDiff(start, GetCurrentTime()));

		// Copy the data to global memory
		start = GetCurrentTime();
		CopyData(local_buf, global_buf, my_range, num_nodes, block_size);

		// Synchronized between processes
		#pragma omp barrier

		// Copy all data from global memory
		for (unsigned i = 0; i < cc_num; i++)
			if (i != self_id)
				CopyData(global_buf, local_buf, range[i], num_nodes, block_size);
		prof->sync_time = TimeAdd(prof->sync_time, TimeDiff(start, GetCurrentTime()));
	}
}

void CalcAPSP(Graph *graph, unsigned block_size) {
	// Concurrency information
	unsigned cc_num = 2; // XXX: We only have a plan for 2 GPU for now.
	omp_set_num_threads(cc_num);

	#pragma omp parallel default(shared)
	{
		int self_id = omp_get_thread_num();

		// Initialize profile data
		Profile prof;
		prof.mem_time = GetZeroTime();
		prof.sync_time = GetZeroTime();

		// Device (GPU) Initialization
		cudaSetDevice(self_id);

		// Allocate buffers
		unsigned nvertices = graph->num_vertices;
		Cost *costs = NewCosts(nvertices * nvertices); // Pinned memory
		Range *range = new Range[cc_num];

		// Copy the original data
		CopyAllData(graph->weights, costs, nvertices, block_size);

		// Allocate memory on GPU
		Cost *costs_on_gpu;
		unsigned data_size = sizeof(Cost) * nvertices * nvertices;
		cudaMalloc((void **) &costs_on_gpu, data_size);

		// Blocked-APSP Algorithm
		unsigned num_rounds = (nvertices % block_size == 0)? nvertices / block_size : nvertices / block_size + 1;
		for (unsigned round_idx = 0; round_idx < num_rounds; round_idx++) {
			unsigned rp1 = round_idx + 1;
			unsigned rr1 = num_rounds - round_idx - 1;

			// Phase 1
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, round_idx, round_idx, 1, 1, &prof);

			// Phase 2
			// Up
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, round_idx, 0, 1, round_idx, &prof);
			// Left
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, 0, round_idx, round_idx, 1, &prof);
			// Right
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, rp1, round_idx, rr1, 1, &prof);
			// Down
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, round_idx, rp1, 1, rr1, &prof);

			// Phase 3
			// Left-Up
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, 0, 0, round_idx, round_idx, &prof);
			// Right-Up
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, rp1, 0, rr1, round_idx, &prof);
			// Left-Down
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, 0, rp1, round_idx, rr1, &prof);
			// Right-Down
			ParallelCalcBlocks(costs, graph->weights, costs_on_gpu, range, cc_num, self_id, nvertices, block_size, round_idx, rp1, rp1, rr1, rr1, &prof);
		}

		// Free memory on GPU
		cudaFree(costs_on_gpu);

		// Deallocate the local buffer
		DeleteCosts(costs);
		delete[] range;

		// Show the profile result
		printf("Thread %d takes %ld ms on memory copying.\n", self_id, TimeToLongInMs(prof.mem_time));
		printf("Thread %d takes %ld ms on synchronizing data.\n", self_id, TimeToLongInMs(prof.sync_time));
	}

}

} // namespace hw4
} // namespace pp