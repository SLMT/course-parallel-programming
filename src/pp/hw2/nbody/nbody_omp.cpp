#include "nbody_omp.hpp"

#include <cstdio>
#include <omp.h>

#include "nbody.hpp"
#include "../../gui.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

void NBodySim(Universe *uni, size_t num_threads, double delta_time, size_t num_steps, double theta, GUI *gui) {
	CelestialBody *bodies = uni->bodies;
	size_t body_count = uni->num_bodies;
	double mass = uni->body_mass;
	double dt = delta_time;

	// Set the number of threads
	omp_set_num_threads(num_threads);

	// Allocate a buffer
	CelestialBody *tmp = new CelestialBody[body_count];

	// Loop for a few steps
	size_t i;
	for (size_t s = 0; s < num_steps; s++) {
		// Calculate new velocities and positions
		#pragma omp parallel for default(shared) private(i)
			for (i = 0; i < body_count; i++) {
				Vec2 total_force = CalculateTotalForce(uni, i);

				// New Velocity
				tmp[i].vel.x = bodies[i].vel.x + total_force.x * dt / mass;
				tmp[i].vel.y = bodies[i].vel.y + total_force.y * dt / mass;
				// New position
				tmp[i].pos.x = bodies[i].pos.x + bodies[i].vel.x * dt;
				tmp[i].pos.y = bodies[i].pos.y + bodies[i].vel.y * dt;
			}

		// Update the states
		#pragma omp parallel for default(shared) private(i)
			for (i = 0; i < body_count; i++)
				bodies[i] = tmp[i];

		// Draw the results
		if (gui != NULL) {
			// Draw all points
			gui->CleanAll();
			for (i = 0; i < body_count; i++) {
				gui->DrawAPoint(bodies[i].pos.x, bodies[i].pos.y);
			}

			// Flush the screen
			gui->Flush();
		}
	}

	// Deallocate the buffer
	delete[] tmp;
}

} // namespace nbody
} // namespace hw2
} // namespace pp
