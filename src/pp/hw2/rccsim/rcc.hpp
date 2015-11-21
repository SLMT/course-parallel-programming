#ifndef PP_HW2_RCCSIM_RCC_H_
#define PP_HW2_RCCSIM_RCC_H_

#include <list>
#include <pthread.h>

namespace pp {
namespace hw2 {
namespace rccsim { // Roller Coaster Car Simulation

class RollerCoasterCar {

public:

	RollerCoasterCar(int capacity);
	~RollerCoasterCar();

	inline int GetCarCapacity() {
		return car_capacity_;
	}

	bool WaifForARide(int id);
	void WaitForCarFull(int *passengers);
	void FinishARide(bool close_car);

private:

	bool is_closed_;
	int car_capacity_;
	std::list<int> *waiting_queue_, *car_seats_;
	pthread_mutex_t car_mutex_;
	pthread_cond_t waiting_cond_, car_full_cond_, playing_cond_;
};

} // namespace rccsim
} // namespace hw2
} // namespace pp

#endif  // PP_HW2_RCCSIM_RCC_H_
