#include "nbody.hpp"

#include <cstdio>

namespace pp {
namespace hw2 {
namespace nbody {

Vec2 Add(Vec2 a, Vec2 b) {
	Vec2 c;
	c.x = a.x + b.x;
	c.y = a.y + b.y;
	return c;
}

Vec2 CalculateTotalForce(Universe *uni, int target) {
	CelestialBody *bodies = uni->bodies;
	double mass = uni->body_mass;
	Vec2 total_force = {0.0, 0.0}, tmp;
	Vec2 dis;

	// Force
	for (size_t i = 0; i < uni->num_bodies; i++) {
		if (i != target) {
			// Calculate the distance
			dis.x = bodies[i].pos.x - bodies[target].pos.x;
			dis.y = bodies[i].pos.y - bodies[target].pos.y;

			// Calculate the force
			Vec2 force = GravitationFormula(mass, mass, dis);
			total_force = Add(force, total_force);
		}
	}

	return total_force;
}

Universe *ReadFromFile(const char *filename) {
	// Open the file
	FILE *file = fopen(filename, "r");

	// Read the first line
	size_t num_bodies;
	fscanf(file, "%lu", &num_bodies);

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
