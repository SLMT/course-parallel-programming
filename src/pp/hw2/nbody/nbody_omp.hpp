#ifndef PP_HW2_NBODY_NBODYOMP_H_
#define PP_HW2_NBODY_NBODYOMP_H_

#include <cstdio>

#include "nbody.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

void NBodySim(Universe *uni, size_t num_threads, double delta_time, size_t num_steps, double theta, XWindowArgs xwin_args);

} // namespace nbody
} // namespace hw2
} // namespace pp

#endif  // PP_HW2_NBODY_NBODYOMP_H_
