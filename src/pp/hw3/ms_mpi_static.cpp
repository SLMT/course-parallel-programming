// This line must be the front of all other includes
#include <mpi.h>

#include "ms_mpi.hpp"
#include "mandelbort_set.hpp"
#include "../timer.hpp"

namespace pp {
namespace hw3 {

void StaticSchedule(int num_threads, ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank) {
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
	Time start = GetCurrentTime();
#ifndef HYBRID
	SeqMSCalculation(start_xs[rank], row_counts[rank], num_colunms, x_scale, y_scale, real_min, imag_min, results);
#else
	Time *th_exe_time = new Time[num_threads];
	int *th_row_counts = new int[num_threads];
	for (unsigned i = 0; i < num_threads; i++) {
		th_exe_time[i] = GetZeroTime();
		th_row_counts[i] = 0;
	}

	OmpMSCalculation(start_xs[rank], row_counts[rank], num_colunms, x_scale, y_scale, real_min, imag_min, results, th_exe_time, th_row_counts);

	for (unsigned i = 0; i < num_threads; i++) {
		printf("Process %d, thread %d took %ld ms to calculate %d rows (%d points).\n", rank, i, TimeToLongInMs(th_exe_time[i]), th_row_counts[i], th_row_counts[i] * num_colunms);
	}

	delete[] th_exe_time;
	delete[] th_row_counts;
#endif
	Time end = GetCurrentTime();

	// Print the execution time
	printf("Process %d took %ld ms to calculate %d rows (%d points).\n", rank, TimeDiffInMs(start, end), row_counts[rank], row_counts[rank] * num_colunms);

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
