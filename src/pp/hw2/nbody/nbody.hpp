#ifndef PP_HW2_NBODY_NBODY_H_
#define PP_HW2_NBODY_NBODY_H_

namespace pp {
namespace hw2 {
namespace nbody {

// ========================
//  N-Body Basic Functions
// ========================

typedef struct {
	double x, y;
} Vec2;

typedef struct {
	Vec2 pos, vel;
} CelestialBody;

typedef struct {
	CelestialBody *bodies;
	int num_bodies;
	double body_mass;
} Universe;

const double kG = 6.674e-011; // Gravitational Constant

Vec2 CalculateTotalForce(Universe *uni, int target);

// ==========================
//  Other Untility Functions
// ==========================

Universe *ReadFromFile(const char *filename);


} // namespace nbody
} // namespace hw2
} // namespace pp

#endif  // PP_HW2_NBODY_NBODY_H_
