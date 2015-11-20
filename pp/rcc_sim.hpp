#ifndef PP_HW2_RCCSIM_RCCSIM_H_
#define PP_HW2_RCCSIM_RCCSIM_H_

#include "rcc.hpp"

namespace pp {
namespace hw2 {
namespace rccsim { // Roller Coaster Car Simulation

typedef struct {
	int passenger_id;
	RollerCoasterCar *car;
} PassengerThreadArgs;

typedef struct {
	RollerCoasterCar *car;
	int playing_time;
	int sim_steps_num;
} CarThreadArgs;

void InitSimluation();

void FinishSimluation();

void *PassengerThread(void *args);

void *CarThread(void *args);

} // namespace rccsim
} // namespace hw2
} // namespace pp

#endif  // PP_HW2_RCCSIM_RCCSIM_H_
