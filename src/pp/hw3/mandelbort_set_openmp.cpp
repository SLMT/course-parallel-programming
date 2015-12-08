#include "mandelbort_set.hpp"

namespace pp {
namespace hw3 {

void ParallelMSCalculation(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, GUI *gui) {
	// For coordination transform
	double x_scale = num_x_points / (real_max - real_min);
	double y_scale = num_y_points / (imag_max - imag_max);

	Complex c;
	for (int x = 0; x < num_threads; x++) {
		for (int y = 0; y < num_y_points; y++) {
			// Map to a complex number
			c.real = x / x_scale + real_min;
			c.imag = y / y_scale + imag_min;

			// Mandelbort Set Check
			int steps = MandelbortSetCheck(c);

			// Draw on the GUI
			XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));
			XDrawPoint (display, window, gc, i, j);
		}
	}
}

} // namespace hw3
} // namespace pp
