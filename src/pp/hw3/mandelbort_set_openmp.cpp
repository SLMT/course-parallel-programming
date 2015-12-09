#include "mandelbort_set.hpp"

#include <omp.h>

namespace pp {
namespace hw3 {

void ParallelMSCalculation(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, GUI *gui) {
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
	unsigned x, y;
#ifndef DYNAMIC
	#pragma omp parallel for default(shared) private(x, y, c, color) schedule(static, 10)
#else
	#pragma omp parallel for default(shared) private(x, y, c, color) schedule(dynamic, 10)
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
		}

	// Draw on the GUI
	if (gui != NULL) {
		for (unsigned x = 0; x < num_x_points; x++) {
			for (unsigned y = 0; y < num_y_points; y++) {
				gui->DrawAPoint(x, y, colors[x * num_y_points + y]);
			}
		}
	}

	delete[] colors;
}

} // namespace hw3
} // namespace pp
