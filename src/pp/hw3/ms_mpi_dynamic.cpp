// This line must be the front of all other includes
#include <mpi.h>

#include <cstring>

#include "ms_mpi.hpp"
#include "mandelbort_set.hpp"

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
		SeqMSCalculation(start, count, num_colunms, x_scale, y_scale, real_min, imag_min, buffer);

		// Request more jobs
		MPI_Send(buffer, count * num_colunms, MPI_UNSIGNED_LONG, kMPIRoot, kTagJobReturnAndRequest, MPI_COMM_WORLD);
		MPI_Recv(&start, 1, MPI_INT, kMPIRoot, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	}

	// Free the resource
	delete[] buffer;
}

} // namespace hw3
} // namespace pp
