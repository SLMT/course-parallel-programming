#include "nbody.hpp"

#include <cstdio>

namespace pp {
namespace hw2 {
namespace nbody {

Vec2 CalculateTotalForce(Universe *uni, int target) {
	Vec2 total_force = {0.0, 0.0}, tmp;

	// Force
	for (size_t i = 0; i < uni->num_bodies; i++) {
		if (i != target) {
			tmp.x = CalculateGravityForce(uni->body_mass, (uni->bodies[target]).pos.x, (uni->bodies[i]).pos.x);
			tmp.y = CalculateGravityForce(uni->body_mass, (uni->bodies[target]).pos.y, (uni->bodies[i]).pos.y);

			total_force.x += tmp.x;
			total_force.y += tmp.y;
		}
	}

	return total_force;
}

Universe *ReadFromFile(const char *filename) {
	// Open the file
	FILE *file = fopen(filename, "r");

	// Read the first line
	size_t num_bodies;
	fscanf(file, "%u", &num_bodies);

	// Read the data of bodies
	CelestialBody *bodies = new CelestialBody[num_bodies];
	for (size_t i = 0; i < num_bodies; i++)
		fscanf(file, "%lf %lf %lf %lf", &(bodies[i].pos.x), &(bodies[i].pos.y), &(bodies[i].vel.x), &(bodies[i].vel.y));

	// Close the file
	fclose(file);

	// Return as an universe
	Universe *uni = new Universe();
	uni->bodies = bodies;
	uni->num_bodies = num_bodies;

	return uni;
}

} // namespace nbody
} // namespace hw2
} // namespace pp
