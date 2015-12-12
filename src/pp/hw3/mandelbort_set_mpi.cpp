// This line must be the front of all other includes
#include <mpi.h>
#include <cstdio>
#include <cstring>
#include <unistd.h>

#include "mandelbort_set.hpp"

namespace pp {
namespace hw3 {

const int kMPIRoot = 0;
const unsigned kRowCountInJob = 10;

// Tags
const int kTagJobRequest = 1;
const int kTagJobReturnAndRequest = 2;
const int kTagNewJob = 3;
const int kTagTerminate = 4;

void StaticSchedule(ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank) {
	// Calculate # of rows for each process
	int rows_per_proc = num_rows / proc_count;
	int left = num_rows % proc_count;
	int *row_counts = new int[proc_count];
	int *buffer_counts = new int[proc_count];
	int *start_xs = new int[proc_count];
	int *offsets = new int[proc_count];
	start_xs[0] = 0;
	offsets[0] = 0;
	for (unsigned r = 0, offset = 0; r < proc_count; r++) {
		if (r < left)
			row_counts[r] = rows_per_proc + 1;
		else
			row_counts[r] = rows_per_proc;
		buffer_counts[r] = row_counts[r] * num_colunms;

		if (r > 0) {
			start_xs[r] = start_xs[r - 1] + row_counts[r - 1];
			offsets[r] = start_xs[r] * num_colunms;
		}
	}

	// Create a local buffer for each process
	ColorHex *results = new ColorHex[buffer_counts[rank]];

	// For coordination transform
	double x_scale = num_rows / (real_max - real_min);
	double y_scale = num_colunms / (imag_max - imag_min);

	// Calculate
	Comp c;
	ColorHex color;
	for (unsigned rx = start_xs[rank], lx = 0; rx < start_xs[rank] + row_counts[rank]; rx++, lx++) {
		for (unsigned y = 0; y < num_colunms; y++) {
			// Map to a complex number
			c.real = rx / x_scale + real_min;
			c.imag = y / y_scale + imag_min;

			// Mandelbort Set Check
			color = MandelbortSetCheck(c);

			// Calculate the color
			color = (color % 256) << 20;
			results[lx * num_colunms + y] = color;
		}
	}

	// The first process collect the results
	MPI_Gatherv(results, row_counts[rank] * num_colunms, MPI_UNSIGNED_LONG,
                colors, buffer_counts, offsets, MPI_UNSIGNED_LONG, kMPIRoot, MPI_COMM_WORLD);

	delete[] row_counts;
	delete[] buffer_counts;
	delete[] start_xs;
	delete[] offsets;
	delete[] results;
}

void DynamicScheduleMaster(ColorHex *colors, int num_rows, int num_colunms, int proc_count) {
	// Allocate a receive buffer
	unsigned buffer_size = kRowCountInJob * num_colunms;
	ColorHex *buffer = new ColorHex[kRowCountInJob * num_colunms];

	// Some variables
	MPI_Status status;
	int next_row_index = 0;
	int *index_per_proc = new int[proc_count];
	int offset, size;

	// Receiving and sending jobs
	while (next_row_index < num_rows) {
		// Wait for a job request or a more job request
		MPI_Recv(buffer, buffer_size, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// If it is a job request
		if (status.MPI_TAG == kTagJobRequest) {
			// Send a new job
			MPI_Send(&next_row_index, 1, MPI_INT, status.MPI_SOURCE, kTagNewJob, MPI_COMM_WORLD);
			index_per_proc[status.MPI_SOURCE] = next_row_index;
		} else if (status.MPI_TAG == kTagJobReturnAndRequest) { // If it is a more job request
			// Calculate the offset and the size
			offset = index_per_proc[status.MPI_SOURCE] * num_colunms;
			size = (index_per_proc[status.MPI_SOURCE] + kRowCountInJob > num_rows)? num_rows - index_per_proc[status.MPI_SOURCE] : kRowCountInJob;
			size = size * num_colunms * sizeof(ColorHex); // in bytes

			// Copy the result to the global buffer
			memcpy(colors + offset, buffer, size);

			// Send a new job
			MPI_Send(&next_row_index, 1, MPI_INT, status.MPI_SOURCE, kTagNewJob, MPI_COMM_WORLD);
			index_per_proc[status.MPI_SOURCE] = next_row_index;
		}

		next_row_index += kRowCountInJob;
	}

	// No more jobs, terminate all slave process
	for (int terminated_count = 0; terminated_count < proc_count - 1; terminated_count++) {
		// Wait for a request
		MPI_Recv(buffer, buffer_size, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// If it is a job request
		if (status.MPI_TAG == kTagJobRequest) {
			// Send a terminate message
			MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, kTagTerminate, MPI_COMM_WORLD);
		} else if (status.MPI_TAG == kTagJobReturnAndRequest) { // If it is a more job request
			// Calculate the offset and the size
			offset = index_per_proc[status.MPI_SOURCE] * num_colunms;
			size = (index_per_proc[status.MPI_SOURCE] + kRowCountInJob > num_rows)? num_rows - index_per_proc[status.MPI_SOURCE] : kRowCountInJob;
			size = size * num_colunms * sizeof(ColorHex); // in bytes

			// Copy the result to the global buffer
			memcpy(colors + offset, buffer, size);

			// Send a terminate message
			MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, kTagTerminate, MPI_COMM_WORLD);
		}
	}

	// Free the resource
	delete[] buffer;
	delete[] index_per_proc;
}

// ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank

void DynamicScheduleSlave(int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int rank) {
	// Allocate a receive buffer
	unsigned buffer_size = kRowCountInJob * num_colunms;
	ColorHex *buffer = new ColorHex[kRowCountInJob * num_colunms];

	// Local variables
	MPI_Status status;
	int start, count;
	double x_scale = num_rows / (real_max - real_min);
	double y_scale = num_colunms / (imag_max - imag_min);

	// Send a job request
	MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, kMPIRoot, kTagJobRequest, MPI_COMM_WORLD);
	MPI_Recv(&start, 1, MPI_INT, kMPIRoot, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	while (status.MPI_TAG == kTagNewJob) {
		count = (start + kRowCountInJob > num_rows)? num_rows - start : kRowCountInJob;

		// Calculate
		Comp c;
		ColorHex color;
		for (unsigned rx = start, lx = 0; rx < start + count; rx++, lx++) {
			for (unsigned y = 0; y < num_colunms; y++) {
				// Map to a complex number
				c.real = rx / x_scale + real_min;
				c.imag = y / y_scale + imag_min;

				// Mandelbort Set Check
				color = MandelbortSetCheck(c);

				// Calculate the color
				color = (color % 256) << 20;
				buffer[lx * num_colunms + y] = color;
			}
		}

		// Request more jobs
		MPI_Send(buffer, count * num_colunms, MPI_UNSIGNED_LONG, kMPIRoot, kTagJobReturnAndRequest, MPI_COMM_WORLD);
		MPI_Recv(&start, 1, MPI_INT, kMPIRoot, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	}

	// Free the resource
	delete[] buffer;
}

void ParallelMSCalculation(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, bool x_enabled) {
	// Get MPI Info
	int proc_count, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Allocate the buffer on the first process
	ColorHex *colors = NULL;
	if (rank == kMPIRoot) {
		colors = new ColorHex[num_x_points * num_y_points];
	}

#ifndef DYNAMIC
	// Static schedule
	StaticSchedule(colors, num_x_points, num_y_points, real_min, real_max, imag_min, imag_max, proc_count, rank);
#else
	// Dynamic schedule
	if (rank == kMPIRoot) {
		DynamicScheduleMaster(colors, num_x_points, num_y_points, proc_count);
	} else {
		DynamicScheduleSlave(num_x_points, num_y_points, real_min, real_max, imag_min, imag_max, rank);
	}
#endif

	// Draw on the GUI by the first process
	if (x_enabled && rank == kMPIRoot) {
		GUI *gui = new GUI(num_x_points, num_y_points);
		for (unsigned x = 0; x < num_x_points; x++) {
			for (unsigned y = 0; y < num_y_points; y++) {
				gui->DrawAPoint(x, y, colors[x * num_y_points + y]);
			}
		}
		gui->Flush();
		sleep(20);
		delete gui;
	}

	// Wait for all process reach the barrier
	MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace hw3
} // namespace pp
