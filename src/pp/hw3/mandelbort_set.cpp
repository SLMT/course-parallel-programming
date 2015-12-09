#include "mandelbort_set.hpp"

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

} // namespace hw3
} // namespace pp
