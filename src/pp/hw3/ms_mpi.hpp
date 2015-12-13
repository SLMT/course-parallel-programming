#ifndef PP_HW3_MSMPI_H_
#define PP_HW3_MSMPI_H_

#include "mandelbort_set.hpp"
#include "../gui.hpp"

namespace pp {
namespace hw3 {

const int kMPIRoot = 0;

// For the dynamic verion
#ifndef HYBRID
	// For normal MPI
	// This value has been proved to be an optimal value under all scenario by experiments.
	const unsigned kRowCountInJob = 1;
#else
	// For hybrid
	// This value has been proved to be a good value when there are a lot of procsses by experiments.
	const unsigned kRowCountInJob = 10;
#endif

// Tags
const int kTagJobRequest = 1;
const int kTagJobReturnAndRequest = 2;
const int kTagNewJob = 3;
const int kTagTerminate = 4;

void StaticSchedule(int num_threads, ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank);
void DynamicScheduleMaster(ColorHex *colors, int num_rows, int num_colunms, int proc_count);
void DynamicScheduleSlave(int num_threads, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int rank);

} // namespace hw3
} // namespace pp

#endif  // PP_HW3_MSMPI_H_
