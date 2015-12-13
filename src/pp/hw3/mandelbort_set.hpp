#ifndef PP_HW3_MANDELBORTSET_H_
#define PP_HW3_MANDELBORTSET_H_

#include "../gui.hpp"
#include "../timer.hpp"

namespace pp {
namespace hw3 {

typedef struct {
	double real, imag;
} Comp;

typedef unsigned long ColorHex;

const unsigned kMaxIteration = 100000;
unsigned MandelbortSetCheck(Comp c);

void SeqMSCalculation(unsigned x_start, unsigned num_rows, unsigned num_cols, double x_scale, double y_scale, double x_min, double y_min, ColorHex *results);
void OmpMSCalculation(unsigned x_start, unsigned num_rows, unsigned num_cols, double x_scale, double y_scale, double x_min, double y_min, ColorHex *results, Time *exe_time, int *cal_rows);

void MSMain(int num_threads, int num_x_points, int num_y_points, double real_min, double real_max, double imag_min, double imag_max, bool x_enabled);

} // namespace hw3
} // namespace pp

#endif  // PP_HW3_MANDELBORTSET_H_
