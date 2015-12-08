#ifndef PP_HW3_MANDELBORTSET_H_
#define PP_HW3_MANDELBORTSET_H_

#include "../gui.hpp"

namespace pp {
namespace hw3 {

typedef struct {
	double real, imag;
} Complex;

const unsigned kMaxIteration = 255;
unsigned MandelbortSetCheck(Complex c);

void ParallelMSCalculation(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, GUI *gui);

typedef unsigned long ColorHex;

} // namespace hw3
} // namespace pp

#endif  // PP_HW3_MANDELBORTSET_H_
