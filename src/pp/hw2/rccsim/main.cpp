#include <cstdio>
#include <cstdlib>
#include <pthread.h>

#include "rcc_sim.hpp"
#include "rcc.hpp"

using pp::hw2::rccsim::RollerCoasterCar;
using pp::hw2::rccsim::PassengerThreadArgs;
using pp::hw2::rccsim::PassengerThread;
using pp::hw2::rccsim::CarThread;
using pp::hw2::rccsim::CarThreadArgs;

int main(int argc, char const *argv[]) {

	// Check arguments
	if (argc < 5) {
		fprintf(stderr, "Insufficient args\n");
		fprintf(stderr, "Usage: %s passengers_num car_capacity playing_time simulation_step_num\nmak", argv[0]);

		return 0;
	}

	// Retrieve arguments
	int passenger_num = (int) strtol(argv[1], NULL, 10);
	int car_capacity = (int) strtol(argv[2], NULL, 10);
	int playing_time = (int) strtol(argv[3], NULL, 10);
	int sim_steps_num = (int) strtol(argv[4], NULL, 10);

	// Create a car instance
	RollerCoasterCar *car = new RollerCoasterCar(car_capacity);

	// Create passenger threads
	pthread_t *pass_threads = new pthread_t[passenger_num];
	PassengerThreadArgs *pass_thread_args = new PassengerThreadArgs[passenger_num];

	for(size_t id = 0; id < passenger_num; id++) {
		pass_thread_args[id].passenger_id = id + 1;
		pass_thread_args[id].car = car;
		pthread_create(pass_threads + id, NULL, PassengerThread, (void *) (pass_thread_args + id));
	}

	// Create the car thread
	pthread_t car_thread;
	CarThreadArgs car_thread_args = {car, playing_time, sim_steps_num};
	pthread_create(&car_thread, NULL, CarThread, (void *) &car_thread_args);

	// Wait for all thread terminates
	for(size_t id = 0; id < passenger_num; id++)
		pthread_join(pass_threads[id], NULL);
	pthread_join(car_thread, NULL);

	// Free all resource
	delete car;
	delete[] pass_threads;
	delete[] pass_thread_args;

	// Exit by pthread
	pthread_exit(NULL);

	return 0;
}
