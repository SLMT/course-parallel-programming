#include <cstdio>

#include "mandelbort_set.hpp"
#include "../gui.hpp"

using pp::GUI;

int main(int argc, char const *argv[]) {
    // Check arguments
	if (argc < 9) {
		fprintf(stderr, "Insufficient args\n");
		fprintf(stderr, "Usage: %s #threads real-min real-max imag-min imag-max #x-points #y-points enable/disable", argv[0]);
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

	// Create a GUI for demonstration
	GUI *gui = NULL;
	if (x_enabled)
		gui = new GUI(num_x_points, num_y_points);

	// Parallel Mandelbort Set Calculation
	ParallelMSCalculation(num_threads, num_x_points, num_y_points, real_min, real_max, imag_min, imag_max, gui);

    return 0;
}
