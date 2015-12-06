#ifndef PP_HW3_MANDELBORTSET_H_
#define PP_HW3_MANDELBORTSET_H_

namespace pp {
namespace hw3 {

typedef struct {
	double real, imag;
} Complex;

int kMaxIteration = 255;

int MandelbortSetCheck(Complex c);

} // namespace hw3
} // namespace pp

#endif  // PP_HW3_MANDELBORTSET_H_
