#include "rcc_sim.hpp"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <unistd.h>

#include "../../timer.hpp"

namespace pp {
namespace hw2 {
namespace rccsim { // Roller Coaster Car Simulation

pthread_mutex_t random_mutex;

void InitSimluation() {
	// Set the seed of random number generator
	srand(time(NULL));

	// Create a mutex
	pthread_mutex_init(&random_mutex, NULL);
}

void FinishSimluation() {
	pthread_mutex_destroy(&random_mutex);
}

useconds_t GetRandomTime() {
	useconds_t num;

	// Protected by mutex
	pthread_mutex_lock(&random_mutex);
	num = (useconds_t) rand();
	pthread_mutex_unlock(&random_mutex);

	return num % 1000;
}

void *PassengerThread(void *args) {
	PassengerThreadArgs *pta = (PassengerThreadArgs *) args;
	bool keep_going = true;

	printf("This is passenger thread no.%d\n", pta->passenger_id);

	while (keep_going) {
		// Wander around
		useconds_t walk_time = GetRandomTime();
		printf("Passenger no.%d wanders around the park for %ld minisecond\n", pta->passenger_id, walk_time);
		usleep(walk_time * 1000);

		// Wait for a ride
		keep_going = !(pta->car->WaifForARide(pta->passenger_id));
	}

	printf("Passenger no.%d leaves.\n", pta->passenger_id);
}

void *CarThread(void *args) {
	CarThreadArgs *cta = (CarThreadArgs *) args;
	int car_capacity = cta->car->GetCarCapacity();
	int *passenger_list = new int[car_capacity];
	char *tmp_str = new char[10];
	char *list_str = new char[car_capacity * 15];
	Time start, now;
	Time wait_start, wait_end, total_wait;

	start = GetCurrentTime();
	total_wait = GetZeroTime();

	printf("This is the car thread. Playing takes %d ms. It will run %d rounds.\n", cta->playing_time, cta->sim_steps_num);

	for (size_t r = 0; r < cta->sim_steps_num; r++) {
		// Check queue
		wait_start = GetCurrentTime();
		cta->car->WaitForCarFull(passenger_list);
		wait_end = GetCurrentTime();
		total_wait = TimeAdd(total_wait, TimeDiff(wait_start, wait_end));

		// Construct string for logging
		list_str[0] = 0; // Clear the string
		sprintf(list_str, "%d", passenger_list[0]);
		for (size_t i = 1; i < car_capacity; i++) {
			sprintf(tmp_str, ", %d", passenger_list[i]);
			strcat(list_str, tmp_str);
		}

		// Print departure log
		now = GetCurrentTime();
		printf("Car departures at %ld millisec. Passengers [%s] are in the car\n", TimeDiffInMs(start, now), list_str);

		// Start playing
		usleep((useconds_t) (cta->playing_time * 1000));

		// Print arrive log
		now = GetCurrentTime();
		printf("Car arrives at %ld millisec. Passengers [%s] get off\n", TimeDiffInMs(start, now), list_str);

		// Release passengers
		if (r < cta->sim_steps_num - 1)
			cta->car->FinishARide(false);
		else {
			cta->car->FinishARide(true);
			printf("The car thread closes the car.\n");
		}
	}

	// Print the average waitting time
	long avg_wait = TimeToLongInMs(total_wait) / cta->sim_steps_num;
	printf("The average waitting time is %ld ms.\n", avg_wait);

	delete[] passenger_list;
	delete[] tmp_str;
	delete[] list_str;
}

} // namespace rccsim
} // namespace hw2
} // namespace pp
