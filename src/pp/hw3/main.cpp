#include <cstdio>

#ifndef OMP
	#include <mpi.h>
#endif

#include "mandelbort_set.hpp"

using pp::hw3::ParallelMSCalculation;

int main(int argc, char *argv[]) {
    // Check arguments
	if (argc < 9) {
		fprintf(stderr, "Insufficient args\n");
		fprintf(stderr, "Usage: %s #threads real-min real-max imag-min imag-max #x-points #y-points enable/disable\n", argv[0]);
		return 0;
	}

	// Retrieve the arguments
    int num_threads, num_x_points, num_y_points;
    double real_min, real_max, imag_min, imag_max;
    bool x_enabled;

    sscanf(argv[1], "%d", &num_threads);
    sscanf(argv[2], "%lf", &real_min);
    sscanf(argv[3], "%lf", &real_max);
    sscanf(argv[4], "%lf", &imag_min);
    sscanf(argv[5], "%lf", &imag_max);
    sscanf(argv[6], "%d", &num_x_points);
    sscanf(argv[7], "%d", &num_y_points);
    x_enabled = (argv[8][0] == 'e')? true : false;

	// Parallel Mandelbort Set Calculation
#ifdef OMP
	ParallelMSCalculation(num_threads, num_x_points, num_y_points, real_min, real_max, imag_min, imag_max, x_enabled);
#else
	// Init MPI
	MPI_Init(&argc, &argv);

	ParallelMSCalculation(num_threads, num_x_points, num_y_points, real_min, real_max, imag_min, imag_max, x_enabled);

	// Finalize MPI
	MPI_Finalize();
#endif

    return 0;
}
