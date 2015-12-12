#include "mandelbort_set.hpp"

#include <cstdio>
#include <cstring>
#include <omp.h>
#include <unistd.h>

#include "../timer.hpp"

namespace pp {
namespace hw3 {

void ParallelMSCalculation(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, bool x_enabled) {
	// Record the start time
	Time start = GetCurrentTime();
	char version[100];

	// For coordination transform
	double x_scale = num_x_points / (real_max - real_min);
	double y_scale = num_y_points / (imag_max - imag_min);

	// For calculation
	Comp c;
	ColorHex color;
	ColorHex *colors = new ColorHex[num_x_points * num_y_points];

	// Set the number of threads
	omp_set_num_threads(num_threads);

	// Calculate
	unsigned x, y, count;
	Time th_start, th_end;

	#pragma omp parallel default(shared) private(x, y, count, c, color, th_start, th_end)
	{
		count = 0;
		th_start = GetCurrentTime();

		// According to my experiments, the best chunk size is 1.
#ifndef DYNAMIC
		strncpy(version, "OpenMP static", 100);
		#pragma omp for schedule(static, 1)
#else
		strncpy(version, "OpenMP dynamic", 100);
		#pragma omp for schedule(dynamic, 1)
#endif
		for (x = 0; x < num_x_points; x++) {
			for (y = 0; y < num_y_points; y++) {
				// Map to a complex number
				c.real = x / x_scale + real_min;
				c.imag = y / y_scale + imag_min;

				// Mandelbort Set Check
				color = MandelbortSetCheck(c);

				// Calculate the color
				color = (color % 256) << 20;
				colors[x * num_y_points + y] = color;
			}
			count++;
			th_end = GetCurrentTime();
		}

		printf("Thread no.%d took %d time to calculate %d rows (%d points).\n", omp_get_thread_num(), TimeDiffInMs(th_start, th_end), count, count * num_y_points);
	}

	// Record the stop time and print the execution time
	Time end = GetCurrentTime();
	printf("%s version took %ld ms for calculation.\n", version, TimeDiffInMs(start, end));

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
}

} // namespace hw3
} // namespace pp
