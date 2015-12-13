#include "mandelbort_set.hpp"

#include <cstdio>
#include <omp.h>

#include "../timer.hpp"

namespace pp {
namespace hw3 {

unsigned MandelbortSetCheck(Comp c) {
	double sq_len;
	Comp z, next_z;

	// Set Z0
	z = c;

	// Mandelbort Set Check
	for (unsigned iter_count = 1; iter_count < kMaxIteration; iter_count++) {
		// Check if the length is in the bound
		sq_len = z.real * z.real + z.imag * z.imag;
		if (sq_len > 4.0)
			return iter_count;

		// Calculate Zk
		next_z.real = z.real * z.real - z.imag * z.imag + c.real;
		next_z.imag = 2 * z.real * z.imag + c.imag;
		z = next_z;
	}

	return kMaxIteration;
}

void SeqMSCalculation(unsigned x_start, unsigned num_rows, unsigned num_cols, double x_scale, double y_scale, double x_min, double y_min, ColorHex *results) {
	Comp c;
	ColorHex color;

	for (unsigned rx = x_start, lx = 0; rx < x_start + num_rows; rx++, lx++) {
		for (unsigned y = 0; y < num_cols; y++) {
			// Map to a complex number
			c.real = rx / x_scale + x_min;
			c.imag = y / y_scale + y_min;

			// Mandelbort Set Check
			color = MandelbortSetCheck(c);

			// Calculate the color
			color = (color % 256) << 20;
			results[lx * num_cols + y] = color;
		}
	}
}

void OmpMSCalculation(unsigned x_start, unsigned num_rows, unsigned num_cols, double x_scale, double y_scale, double x_min, double y_min, ColorHex *results, Time *exe_time, int *cal_rows) {
	Comp c;
	ColorHex color;
	unsigned x, y;
	Time tstart, tend;
	int count;

	#pragma omp parallel default(shared) private(c, color, x, y, tstart, tend, count)
	{
		count = 0;
		tstart = GetCurrentTime();

#ifndef DYNAMIC
		#pragma omp for schedule(static, 1)
#else
		#pragma omp for schedule(dynamic, 1)
#endif
		for (x = 0; x < num_rows; x++) {
			for (y = 0; y < num_cols; y++) {
				// Map to a complex number
				c.real = (x + x_start) / x_scale + x_min;
				c.imag = y / y_scale + y_min;

				// Mandelbort Set Check
				color = MandelbortSetCheck(c);

				// Calculate the color
				color = (color % 256) << 20;
				results[x * num_cols + y] = color;
			}
			count++;
			tend = GetCurrentTime();
		}

		// For experiments
		exe_time[omp_get_thread_num()] = TimeAdd(exe_time[omp_get_thread_num()], TimeDiff(tstart, tend));
		cal_rows[omp_get_thread_num()] += count;
	}
}

} // namespace hw3
} // namespace pp
