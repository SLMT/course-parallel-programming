#ifndef PP_HW3_MSMPI_H_
#define PP_HW3_MSMPI_H_

#include "mandelbort_set.hpp"
#include "../gui.hpp"

namespace pp {
namespace hw3 {

const int kMPIRoot = 0;

// For the dynamic verion
const unsigned kRowCountInJob = 10;

// Tags
const int kTagJobRequest = 1;
const int kTagJobReturnAndRequest = 2;
const int kTagNewJob = 3;
const int kTagTerminate = 4;

void StaticSchedule(ColorHex *colors, int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int proc_count, int rank);
void DynamicScheduleMaster(ColorHex *colors, int num_rows, int num_colunms, int proc_count);
void DynamicScheduleSlave(int num_rows, int num_colunms, double real_min, double real_max, double imag_min, double imag_max, int rank);

} // namespace hw3
} // namespace pp

#endif  // PP_HW3_MSMPI_H_
