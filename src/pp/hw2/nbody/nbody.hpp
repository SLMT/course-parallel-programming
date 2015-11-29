#ifndef PP_HW2_NBODY_NBODY_H_
#define PP_HW2_NBODY_NBODY_H_

#include <cmath>

namespace pp {
namespace hw2 {
namespace nbody {

// ========================
//  N-Body Basic Functions
// ========================

typedef struct {
	double x, y;
} Vec2;

Vec2 Add(Vec2 a, Vec2 b);

typedef struct {
	Vec2 pos, vel;
} CelestialBody;

typedef struct {
	CelestialBody *bodies;
	int num_bodies;
	double body_mass;
} Universe;

const double kG = 6.674e-011; // Gravitational Constant
const double kMinDis = 1.0e-004;

inline Vec2 GravitationFormula(double m1, double m2, Vec2 dis) {
	Vec2 force;
	double dis_total = sqrt(dis.x * dis.x + dis.y * dis.y);
	/*
	double force_value;
	if (dis_total < kMinDis) // Give a minimun value to the distance
		force_value = (kG * m1 * m2) / (kMinDis * kMinDis);
	else
		force_value = (kG * m1 * m2) / (dis_total * dis_total);
	*/
	double force_value = (kG * m1 * m2) / (dis_total * dis_total);
	force.x = force_value * dis.x / dis_total;
	force.y = force_value * dis.y / dis_total;
	return force;
}

Vec2 CalculateTotalForce(Universe *uni, int target);

// ==========================
//  Other Untility Functions
// ==========================

Universe *ReadFromFile(const char *filename);


} // namespace nbody
} // namespace hw2
} // namespace pp

#endif  // PP_HW2_NBODY_NBODY_H_
