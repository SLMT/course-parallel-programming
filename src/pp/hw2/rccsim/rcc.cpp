#include "rcc.hpp"

#include <cstdio>
#include <pthread.h>

namespace pp {
namespace hw2 {
namespace rccsim { // Roller Coaster Car Simulation

RollerCoasterCar::RollerCoasterCar(int capacity) {
	is_closed_ = false;
	car_capacity_ = capacity;
	waiting_queue_ = new std::list<int>();
	car_seats_ = new std::list<int>();
	pthread_mutex_init(&car_mutex_, NULL);
	pthread_cond_init(&waiting_cond_, NULL);
	pthread_cond_init(&car_full_cond_, NULL);
	pthread_cond_init(&playing_cond_, NULL);
}

RollerCoasterCar::~RollerCoasterCar() {
	delete waiting_queue_;
	delete car_seats_;
	pthread_mutex_destroy(&car_mutex_);
	pthread_cond_destroy(&waiting_cond_);
	pthread_cond_destroy(&car_full_cond_);
	pthread_cond_destroy(&playing_cond_);
}

bool RollerCoasterCar::WaifForARide(int id) {
	bool is_closed;

	pthread_mutex_lock(&car_mutex_);

	printf("Passenger no.%d gets in queue for a ride\n", id);

	// Check if the car is full
	if (car_seats_->size() >= car_capacity_) {

		// Add to the waiting queue
		waiting_queue_->push_back(id);

		// Wait until there is a seat for it and the thread must be the head of the waiting queue
		while (!(car_seats_->size() < car_capacity_ &&
				waiting_queue_->front() == id))
			pthread_cond_wait(&waiting_cond_, &car_mutex_);

		// Remove from the queue
		waiting_queue_->pop_front();

		// Notify other thread to get the seats
		pthread_cond_broadcast(&waiting_cond_);
	}

	is_closed = is_closed_;

	if (!is_closed) {
		// Get in the car
		car_seats_->push_back(id);

		if (car_seats_->size() >= car_capacity_) {
			// Notify the car thread to start
			pthread_cond_signal(&car_full_cond_);
		}

		// Wait for the ride to departure
		pthread_cond_wait(&playing_cond_, &car_mutex_);
	}

	pthread_mutex_unlock(&car_mutex_);

	return is_closed;
}

void RollerCoasterCar::WaitForCarFull(int *passengers) {
	pthread_mutex_lock(&car_mutex_);

	// Wait for the car full
	while (car_seats_->size() < car_capacity_) {
		// Wait for the car full
		pthread_cond_wait(&car_full_cond_, &car_mutex_);
	}

	// Return the list
	size_t i = 0;
	for (std::list<int>::iterator it = car_seats_->begin();
			it != car_seats_->end(); it++, i++)
		passengers[i] = *it;

	pthread_mutex_unlock(&car_mutex_);
}

void RollerCoasterCar::FinishARide(bool close_car) {
	pthread_mutex_lock(&car_mutex_);

	// Record the car is closed
	is_closed_ = close_car;

	// Notify the passengers to leave
	car_seats_->clear();
	pthread_cond_broadcast(&playing_cond_);

	// Notify the passengers in the waiting queue waking up
	pthread_cond_broadcast(&waiting_cond_);

	pthread_mutex_unlock(&car_mutex_);
}

} // namespace rccsim
} // namespace hw2
} // namespace pp
