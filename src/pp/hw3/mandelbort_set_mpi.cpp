#include "mandelbort_set.hpp"

namespace pp {
namespace hw3 {

const int kMPIRoot = 0;

void StaticSchedule(ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank) {
	// For coordination transform
	double x_scale = num_x_points / (real_max - real_min);
	double y_scale = num_y_points / (imag_max - imag_max);

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

	// Calculate
	Complex c;
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
                colors, buffer_counts, offsets, MPI_UNSIGNED_LONG, kMPIRoot, MPI_COMM_WORLD)

	delete[] row_counts;
	delete[] buffer_counts;
	delete[] start_xs;
	delete[] offsets;
	delete[] results;
}

void ParallelMSCalculation(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, GUI *gui) {
	// Get MPI Info
	int proc_count, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// Allocate the buffer on the first process
	ColorHex *colors = NULL;
	if (rank == kMPIRoot)
		colors = new ColorHex[num_x_points * num_y_points];

#ifndef DYNAMIC
	// Static schedule
	StaticSchedule(colors, num_x_points, num_y_points, real_min, real_max, imag_min, imag_max, proc_count, rank);
#else
	// TODO: Dynamic schedule
#endif

	// Draw on the GUI by the first process
	if (gui != NULL && rank == kMPIRoot) {
		for (unsigned x = 0; x < num_x_points; x++) {
			for (unsigned y = 0; y < num_y_points; y++) {
				gui->DrawAPoint(x, y, colors[x * num_y_points + y]);
			}
		}
	}

	// Wait for all process reach the barrier
	MPI_Barrier(MPI_COMM_WORLD);
}

} // namespace hw3
} // namespace pp
