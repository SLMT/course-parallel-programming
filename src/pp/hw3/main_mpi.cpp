// This line must be the front of all other includes
#include <mpi.h>

#include <cstdio>
#include <unistd.h>

#include "mandelbort_set.hpp"
#include "ms_mpi.hpp"

namespace pp {
namespace hw3 {

void MSMain(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, bool x_enabled) {
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
