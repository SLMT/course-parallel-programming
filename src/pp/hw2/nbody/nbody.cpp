#include "nbody.hpp"

#include <cstdio>

namespace pp {
namespace hw2 {
namespace nbody {

void CalculateNewVelocity(Universe *uni, int target, double delta_time) {
	Vec2 total_force = {0.0, 0.0}, tmp;

	// Force
	for (size_t i = 0; i < uni->num_body; i++) {
		tmp.x = CalculateGravityForce(uni->body_mass, (uni->bodies[target]).pos.x, (uni->bodies[i]).pos.x);
		tmp.y = CalculateGravityForce(uni->body_mass, (uni->bodies[target]).pos.y, (uni->bodies[i]).pos.y);

		total_force.x += tmp.x;
		total_force.y += tmp.y;
	}

	// New Velocity
	(uni->bodies[target]).vel.x += total_force.x * delta_time / uni->body_mass;
	(uni->bodies[target]).vel.y += total_force.y * delta_time / uni->body_mass;
}

} // namespace nbody
} // namespace hw2
} // namespace pp

int main(void) {
	pp::hw2::nbody::Universe uni;
	uni.bodies = new pp::hw2::nbody::CelestialBody[3];
	uni.num_body = 3;
	uni.body_mass = 1;

	for (size_t i = 0; i < 3; i++) {
		uni.bodies[i].pos.x = 0.0;
		uni.bodies[i].pos.y = 0.0;
		uni.bodies[i].vel.x = 0.0;
		uni.bodies[i].vel.y = 0.0;
	}

	uni.bodies[1].pos.x = 1.0;
	uni.bodies[1].pos.y = 1.0;
	uni.bodies[2].pos.x = 2.0;
	uni.bodies[2].pos.y = 2.0;

	pp::hw2::nbody::CalculateNewVelocity(&uni, 1, 1.0);

	printf("x: %lf, y: %lf\n", uni.bodies[1].vel.x, uni.bodies[1].vel.y);

	return 0;
}
