#include "nbody.hpp"

#include <cstdio>
#include <cmath>

namespace pp {
namespace hw2 {
namespace nbody {

Vec2 CalculateTotalForce(Universe *uni, int target) {
	CelestialBody *bodies = uni->bodies;
	double mass = uni->body_mass;
	Vec2 total_force = {0.0, 0.0}, tmp;
	double dis_x, dis_y, dis_total;
	double force;

	// Force
	for (size_t i = 0; i < uni->num_bodies; i++) {
		if (i != target) {
			// Calculate the distance
			dis_x = bodies[i].pos.x - bodies[target].pos.x;
			dis_y = bodies[i].pos.y - bodies[target].pos.y;
			dis_total = sqrt(dis_x * dis_x + dis_y * dis_y);

			// Calculate the force
			force = (kG * mass * mass) / (dis_total * dis_total);
			total_force.x += force * dis_x / dis_total;
			total_force.y += force * dis_y / dis_total;
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
