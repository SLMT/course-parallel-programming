#include "apsp.hpp"

#include <mpi.h>

#include "block_calculation.hpp"
#include "../timer.hpp"

namespace pp {
namespace hw4 {

typedef struct {
	unsigned x_start, y_start, x_len, y_len;
} Range;

typedef struct {
	unsigned *send_buf;
	unsigned send_count;
	unsigned *recv_buf;
	unsigned recv_count;
} Diff;

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

void GenerateDiff(Cost *before, Cost *after, unsigned self_id, Range *range, unsigned num_nodes, unsigned block_size, Diff *diff) {
	unsigned count = 0;
	Range r = range[self_id];
	unsigned *diffs = diff->send_buf;

	// Compare only inside the range
	for (unsigned bx = r.x_start; bx < r.x_start + r.x_len; bx++) {
		for (unsigned by = r.y_start; by < r.y_start + r.y_len; by++) {
			for (unsigned i = bx * block_size; i < (bx + 1) * block_size && i < num_nodes; i++) {
				for (unsigned j = by * block_size; j < (by + 1) * block_size && j < num_nodes; j++) {
					if (before[i * num_nodes + j] != after[i * num_nodes + j]) {
						diffs[count++] = i;
						diffs[count++] = j;
						diffs[count++] = after[i * num_nodes + j];

						// Copy the data in the same time
						before[i * num_nodes + j] = after[i * num_nodes + j];
					}
				}
			}
		}
	}

	diff->send_count = count;
}

void ApplyDiff(Cost *costs, Diff *diff, unsigned num_nodes) {
	unsigned x, y;
	unsigned *diffs = diff->recv_buf;

	for (unsigned i = 0; i < diff->recv_count;) {
		x = diffs[i++];
		y = diffs[i++];
		costs[x * num_nodes + y] = diffs[i++];
	}
}

const int kTagDataCount = 1;
const int kTagData = 2;

void SendData(Diff *diff, unsigned cc_num, unsigned self_id) {
	MPI_Request req;

	// Send it out to each process
	for (unsigned id = 0; id < cc_num; id++)
		if (id != self_id) {
			MPI_Isend(&(diff->send_count), 1, MPI_UNSIGNED, id, kTagDataCount, MPI_COMM_WORLD, &req);
			if (diff->send_count > 0)
				MPI_Isend(diff->send_buf, diff->send_count, MPI_UNSIGNED, id, kTagData, MPI_COMM_WORLD, &req);
		}
}

void RecvData(Cost *costs, Diff *diff, unsigned cc_num, unsigned self_id, unsigned num_nodes) {
	MPI_Status status;

	for (unsigned id = 0; id < cc_num; id++)
		if (id != self_id) {
			// Receive data
			MPI_Recv(&(diff->recv_count), 1, MPI_UNSIGNED, id, kTagDataCount, MPI_COMM_WORLD, &status);

			if (diff->recv_count > 0) {
				MPI_Recv(diff->recv_buf, diff->recv_count, MPI_UNSIGNED, id, kTagData, MPI_COMM_WORLD, &status);
				ApplyDiff(costs, diff, num_nodes);
			}
		}
}

Time mem_time, com_time;

void ParallelCalcBlocks(Cost *local_data, Cost *result, Cost *gpu_data, Diff *diff, Range *range, unsigned cc_num, unsigned self_id, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned bx, unsigned by, unsigned bxlen, unsigned bylen) {
	Time start;

	if (bxlen != 0 && bylen != 0) {
		// Determine a range for calculation
		DetermineCalculateRange(range, bx, by, bxlen, bylen);
		Range my_range = range[self_id];

		// Copy the data from Host to Device
		start = GetCurrentTime();
		unsigned data_size = sizeof(Cost) * num_nodes * num_nodes;
		cudaMemcpy(gpu_data, local_data, data_size, cudaMemcpyHostToDevice);
		mem_time = TimeAdd(mem_time, TimeDiff(start, GetCurrentTime()));

		//printf("Process %d in round %u range (%u, %u, %u, %u)\n", self_id, round_idx, my_range.x_start, my_range.y_start, my_range.x_len, my_range.y_len);

		// Calculate the blocks in the range
		CalcBlocks(gpu_data, num_nodes, block_size, round_idx, my_range.x_start, my_range.y_start, my_range.x_len, my_range.y_len);
		// Wait for complete
		cudaThreadSynchronize();

		// Copy the data from Device to Host
		start = GetCurrentTime();
		cudaMemcpy(result, gpu_data, data_size, cudaMemcpyDeviceToHost);
		mem_time = TimeAdd(mem_time, TimeDiff(start, GetCurrentTime()));

		start = GetCurrentTime();
		// Generate difference
		GenerateDiff(local_data, result, self_id, range, num_nodes, block_size, diff);

		// Send the data in its range to other processes (threads)
		SendData(diff, cc_num, self_id);

		// Receive the data from other processes (threads)
		RecvData(local_data, diff, cc_num, self_id, num_nodes);

		// Synchronized between processes
		MPI_Barrier(MPI_COMM_WORLD);
		com_time = TimeAdd(com_time, TimeDiff(start, GetCurrentTime()));
	}
}

void CalcAPSP(Graph *graph, unsigned block_size) {
	Time start;
	MPI_Status status;
	mem_time = GetZeroTime();
	com_time = GetZeroTime();

	// Concurrency information
	int self_id;
	unsigned cc_num = 2; // XXX: We only have a plan for 2 GPU for now.
	MPI_Comm_rank(MPI_COMM_WORLD, &self_id);

	// Initial send all data to other processes
	start = GetCurrentTime();
	unsigned nvertices;
	Cost *costs;
	if (self_id == 0) {
		nvertices = graph->num_vertices;
		costs = graph->weights;

		for (unsigned id = 1; id < cc_num; id++) {
			MPI_Send(&nvertices, 1, MPI_UNSIGNED, id, kTagDataCount, MPI_COMM_WORLD);
			MPI_Send(costs, nvertices * nvertices, MPI_UNSIGNED, id, kTagData, MPI_COMM_WORLD);
		}
	} else {
		MPI_Recv(&nvertices, 1, MPI_UNSIGNED, 0, kTagDataCount, MPI_COMM_WORLD, &status);

		costs = NewCosts(nvertices * nvertices); // Pinned memory

		MPI_Recv(costs, nvertices * nvertices, MPI_UNSIGNED, 0, kTagData, MPI_COMM_WORLD, &status);

		graph = new Graph();
		graph->num_vertices = nvertices;
		graph->weights = costs;
	}
	com_time = TimeAdd(com_time, TimeDiff(start, GetCurrentTime()));

	// Allocate some other buffers
	Cost *result_buf = NewCosts(nvertices * nvertices); // Pinned memory
	Diff diff;
	diff.send_buf = new unsigned[3 * nvertices * nvertices];
	diff.recv_buf = new unsigned[3 * nvertices * nvertices];
	Range *range = new Range[cc_num];

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
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, round_idx, round_idx, 1, 1);

		// Phase 2
		// Up
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, round_idx, 0, 1, round_idx);
		// Left
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, 0, round_idx, round_idx, 1);
		// Right
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, rp1, round_idx, rr1, 1);
		// Down
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, round_idx, rp1, 1, rr1);

		// Phase 3
		// Left-Up
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, 0, 0, round_idx, round_idx);
		// Right-Up
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, rp1, 0, rr1, round_idx);
		// Left-Down
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, 0, rp1, round_idx, rr1);
		// Right-Down
		ParallelCalcBlocks(costs, result_buf, costs_on_gpu, &diff, range, cc_num, self_id, nvertices, block_size, round_idx, rp1, rp1, rr1, rr1);
	}

	// Free memory on GPU
	cudaFree(costs_on_gpu);

	// Deallocate the local buffer
	if (self_id > 0)
		DeleteCosts(costs);
	DeleteCosts(result_buf);
	delete[] diff.send_buf;
	delete[] diff.recv_buf;
	delete[] range;

	// Show the profile result
	printf("Process %d takes %ld ms on memory copying.\n", self_id, TimeToLongInMs(mem_time));
	printf("Process %d takes %ld ms on communication.\n", self_id, TimeToLongInMs(com_time));
}

} // namespace hw4
} // namespace pp
