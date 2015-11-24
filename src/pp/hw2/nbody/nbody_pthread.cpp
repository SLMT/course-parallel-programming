#include "nbody_pthread.hpp"

#include <cstdio>
#include <pthread.h>

#include "nbody.hpp"
#include "gui.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

typedef struct {
	// Thread info
	size_t thread_id;
	size_t num_threads;
	pthread_barrier_t *barrier;

	// N-Body parameters
	Universe *uni;
	double delta_time;
	size_t num_steps;
} PThreadSimArgs;

typedef struct {
	pthread_barrier_t *barrier;
	size_t num_steps;
	GUI *gui;
	Universe *uni;
} PThreadGUIArgs;

void *PThreadSimTask(void *args) {
	PThreadSimArgs *tas = (PThreadSimArgs *) args;
	int tid = tas->thread_id;
	double dt = tas->delta_time;
	CelestialBody *bodies = tas->uni->bodies;
	int mass = tas->uni->body_mass;
	int thread_count = tas->num_threads;
	int body_count = tas->uni->num_bodies;

	// Allocate a buffer
	CelestialBody *tmp = new CelestialBody[body_count];

	// Loop for a few steps
	for (size_t s = 0; s < tas->num_steps; s++) {
		// Calculate new velocities and positions
		for (size_t i = 0; i < body_count; i++) {
			if (i % thread_count == tid) {
				Vec2 total_force = CalculateTotalForce(tas->uni, i);

				// New Velocity
				tmp[i].vel.x = bodies[i].vel.x + total_force.x * dt / mass;
				tmp[i].vel.y = bodies[i].vel.y + total_force.y * dt / mass;
				// New position
				tmp[i].pos.x = bodies[i].pos.x + bodies[i].vel.x * dt;
				tmp[i].pos.y = bodies[i].pos.y + bodies[i].vel.y * dt;
			}
		}

		// A thread Barrier
		pthread_barrier_wait(tas->barrier);

		// Update the states
		for (size_t i = 0; i < body_count; i++)
			if (i % thread_count == tid)
				bodies[i] = tmp[i];

		// Another thread Barrier
		pthread_barrier_wait(tas->barrier);
	}

	// Deallocate the buffer
	delete[] tmp;
}

void *PThreadGUITask(void *args) {
	PThreadGUIArgs *tgas = (PThreadGUIArgs*) args;
	GUI *gui = tgas->gui;
	CelestialBody *bodies = tgas->uni->bodies;

	// Loop for a few steps
	for (size_t s = 0; s < tgas->num_steps; s++) {
		// Draw all bodies
		gui->CleanAll();
		for (size_t i = 0; i < tgas->uni->num_bodies; i++) {
			gui->DrawAPoint(bodies[i].pos.x, bodies[i].pos.y);
		}

		// Flush the screen
		// XXX: Maybe I can put this after barrier
		gui->Flush();

		// A thread Barrier
		pthread_barrier_wait(tgas->barrier);

		// Another thread Barrier
		pthread_barrier_wait(tgas->barrier);
	}
}

void NBodySim(Universe *uni, size_t num_threads, double delta_time, size_t num_steps, double theta, GUI *gui) {
	// Initialize pthread barrier
	pthread_barrier_t barrier;
	if (gui != NULL)
		pthread_barrier_init(&barrier, NULL, num_threads + 1);
	else
		pthread_barrier_init(&barrier, NULL, num_threads);

	// Create simulation threads
	pthread_t *threads = new pthread_t[num_threads];
	PThreadSimArgs *thread_args = new PThreadSimArgs[num_threads];

	for(size_t id = 0; id < num_threads; id++) {
		thread_args[id].thread_id = id;
		thread_args[id].num_threads = num_threads;
		thread_args[id].barrier = &barrier;
		thread_args[id].uni = uni;
		thread_args[id].delta_time = delta_time;
		thread_args[id].num_steps = num_steps;
		pthread_create(threads + id, NULL, PThreadSimTask, (void *) (thread_args + id));
	}

	// Create a GUI thread
	pthread_t gui_thread;
	PThreadGUIArgs gui_args;
	if (gui != NULL) {
		gui_args.gui = gui;
		gui_args.uni = uni;
		gui_args.barrier = &barrier;
		gui_args.num_steps = num_steps;
		pthread_create(&gui_thread, NULL, PThreadGUITask, (void *) &gui_args);
	}

	// Wait for all thread terminates
	for(size_t id = 0; id < num_threads; id++)
		pthread_join(threads[id], NULL);
	if (gui != NULL)
		pthread_join(gui_thread, NULL);

	// Free all resource
	pthread_barrier_destroy(&barrier);
	delete[] threads;
	delete[] thread_args;

	// Exit by pthread
	pthread_exit(NULL);
}

} // namespace nbody
} // namespace hw2
} // namespace pp
