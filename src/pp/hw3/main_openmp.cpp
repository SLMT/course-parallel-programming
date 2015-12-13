#include "mandelbort_set.hpp"

#include <cstdio>
#include <cstring>
#include <omp.h>
#include <unistd.h>

#include "../timer.hpp"

namespace pp {
namespace hw3 {

void MSMain(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, bool x_enabled) {
	// Welcome message
#ifndef DYNAMIC
	printf("== Mandelbort Set OpenMP Static Version ==\n");
#else
	printf("== Mandelbort Set OpenMP Dynamic Version ==\n");
#endif

	// For coordination transform
	double x_scale = num_x_points / (real_max - real_min);
	double y_scale = num_y_points / (imag_max - imag_min);

	// Allocate a few buffer
	ColorHex *colors = new ColorHex[num_x_points * num_y_points];
	Time *exe_time = new Time[num_threads];
	int *row_counts = new int[num_threads];
	for (unsigned i = 0; i < num_threads; i++) {
		exe_time[i] = GetZeroTime();
		row_counts[i] = 0;
	}

	// Set the number of threads
	omp_set_num_threads(num_threads);

	// Calculate the Mandelbort Set
	Time start = GetCurrentTime();
	OmpMSCalculation(0, num_x_points, num_y_points, x_scale, y_scale, real_min, imag_min, colors, exe_time, row_counts);
	Time end = GetCurrentTime();

	// Print the execution time
	printf("It took %ld ms for calculation.\n", TimeDiffInMs(start, end));

	// Draw on the GUI
	if (x_enabled) {
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

	delete[] colors;
	delete[] exe_time;
	delete[] row_counts;
}

} // namespace hw3
} // namespace pp
