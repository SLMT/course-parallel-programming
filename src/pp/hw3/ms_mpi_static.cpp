// This line must be the front of all other includes
#include <mpi.h>

#include "ms_mpi.hpp"
#include "mandelbort_set.hpp"

namespace pp {
namespace hw3 {

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
	SeqMSCalculation(start_xs[rank], row_counts[rank], num_colunms, x_scale, y_scale, real_min, imag_min, results);

	// The first process collect the results
	MPI_Gatherv(results, row_counts[rank] * num_colunms, MPI_UNSIGNED_LONG,
                colors, buffer_counts, offsets, MPI_UNSIGNED_LONG, kMPIRoot, MPI_COMM_WORLD);

	delete[] row_counts;
	delete[] buffer_counts;
	delete[] start_xs;
	delete[] offsets;
	delete[] results;
}

} // namespace hw3
} // namespace pp
