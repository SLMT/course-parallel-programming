// This line must be the front of all other includes
#include <mpi.h>

#include <cstring>

#include "ms_mpi.hpp"
#include "mandelbort_set.hpp"
#include "../timer.hpp"

namespace pp {
namespace hw3 {

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

		// Handle the returned results
		if (status.MPI_TAG == kTagJobReturnAndRequest) {
			// Calculate the offset and the size
			offset = index_per_proc[status.MPI_SOURCE] * num_colunms;
			size = (index_per_proc[status.MPI_SOURCE] + kRowCountInJob > num_rows)? num_rows - index_per_proc[status.MPI_SOURCE] : kRowCountInJob;
			size = size * num_colunms * sizeof(ColorHex); // in bytes

			// Copy the result to the global buffer
			memcpy(colors + offset, buffer, size);
		}

		// Send a new job
		MPI_Send(&next_row_index, 1, MPI_INT, status.MPI_SOURCE, kTagNewJob, MPI_COMM_WORLD);
		index_per_proc[status.MPI_SOURCE] = next_row_index;
		next_row_index += kRowCountInJob;
	}

	// No more jobs, terminate all slave process
	for (int terminated_count = 0; terminated_count < proc_count - 1; terminated_count++) {
		// Wait for a request
		MPI_Recv(buffer, buffer_size, MPI_UNSIGNED_LONG, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		// If it is a job request
		if (status.MPI_TAG == kTagJobReturnAndRequest) { // If it is a more job request
			// Calculate the offset and the size
			offset = index_per_proc[status.MPI_SOURCE] * num_colunms;
			size = (index_per_proc[status.MPI_SOURCE] + kRowCountInJob > num_rows)? num_rows - index_per_proc[status.MPI_SOURCE] : kRowCountInJob;
			size = size * num_colunms * sizeof(ColorHex); // in bytes

			// Copy the result to the global buffer
			memcpy(colors + offset, buffer, size);
		}

		// Send a terminate message
		MPI_Send(NULL, 0, MPI_INT, status.MPI_SOURCE, kTagTerminate, MPI_COMM_WORLD);
	}

	// Free the resource
	delete[] buffer;
	delete[] index_per_proc;
}

// ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank

void DynamicScheduleSlave(int num_threads, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int rank) {
	// Allocate a receive buffer
	unsigned buffer_size = kRowCountInJob * num_colunms;
	ColorHex *buffer = new ColorHex[kRowCountInJob * num_colunms];

	// Local variables
	MPI_Status status;
	int start, count;
	double x_scale = num_rows / (real_max - real_min);
	double y_scale = num_colunms / (imag_max - imag_min);

	// Record the start time
	Time ts = GetCurrentTime();
	int total_rows = 0;
#ifdef HYBRID
	Time *th_exe_time = new Time[num_threads];
	int *th_row_counts = new int[num_threads];
	for (unsigned i = 0; i < num_threads; i++) {
		th_exe_time[i] = GetZeroTime();
		th_row_counts[i] = 0;
	}
#endif

	// Send a job request
	MPI_Send(NULL, 0, MPI_UNSIGNED_LONG, kMPIRoot, kTagJobRequest, MPI_COMM_WORLD);
	MPI_Recv(&start, 1, MPI_INT, kMPIRoot, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

	while (status.MPI_TAG == kTagNewJob) {
		count = (start + kRowCountInJob > num_rows)? num_rows - start : kRowCountInJob;
		total_rows += count;

		// Calculate
#ifndef HYBRID
		SeqMSCalculation(start, count, num_colunms, x_scale, y_scale, real_min, imag_min, buffer);
#else
		OmpMSCalculation(start, count, num_colunms, x_scale, y_scale, real_min, imag_min, buffer, th_exe_time, th_row_counts);
#endif

		// Request more jobs
		MPI_Send(buffer, count * num_colunms, MPI_UNSIGNED_LONG, kMPIRoot, kTagJobReturnAndRequest, MPI_COMM_WORLD);
		MPI_Recv(&start, 1, MPI_INT, kMPIRoot, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	}

	// Print the execution time
#ifdef HYBRID
	for (unsigned i = 0; i < num_threads; i++) {
		printf("Process %d, thread %d took %ld ms to calculate %d rows (%d points).\n", rank, i, TimeToLongInMs(th_exe_time[i]), th_row_counts[i], th_row_counts[i] * num_colunms);
	}

	delete[] th_exe_time;
	delete[] th_row_counts;
#endif

	Time te = GetCurrentTime();
	printf("Process %d took %ld ms to calculate %d rows (%d points).\n", rank, TimeDiffInMs(ts, te), total_rows, total_rows * num_colunms);

	// Free the resource
	delete[] buffer;
}

} // namespace hw3
} // namespace pp
