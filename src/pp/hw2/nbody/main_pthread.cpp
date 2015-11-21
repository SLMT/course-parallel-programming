#include <cstdio>
#include <cstdlib>

#include "nbody.hpp"
#include "nbody_pthread.hpp"

using pp::hw2::nbody::Universe;
using pp::hw2::nbody::XWindowArgs;
using pp::hw2::nbody::ReadFromFile;
using pp::hw2::nbody::NBodyPThreadImpl;

int main(int argc, char const *argv[]) {

	// Check arguments
	if (argc < 5) {
		fprintf(stderr, "Insufficient args\n");
		fprintf(stderr, "Usage: %s #threads m T t FILE theta enable/disable xmin ymin length Length", argv[0]);
		return 0;
	}

	// Retrieve the arguments
	size_t num_threads, num_steps;
	int mass;
	double delta_time, theta, x_min, y_min, coord_len, win_len;
	const char *filename;
	bool x_enabled;

	num_threads = (size_t) strtol(argv[1], NULL, 10);
	mass = (int) strtol(argv[2], NULL, 10);
	num_steps = (size_t) strtol(argv[3], NULL, 10);
	sscanf(argv[4], "%lf", &delta_time);
	filename = argv[5];
	sscanf(argv[6], "%lf", &theta);
	x_enabled = (argv[7][0] == 'e')? true : false;
	sscanf(argv[8], "%lf", &x_min);
	sscanf(argv[9], "%lf", &y_min);
	sscanf(argv[10], "%lf", &coord_len);
	sscanf(argv[11], "%lf", &win_len);

	// Read the input file
	Universe *uni = ReadFromFile(filename);

	// Start running
	XWindowArgs x_win_args = {x_enabled, x_min, y_min, coord_len, win_len};
	NBodyPThreadImpl(uni, num_threads, delta_time, num_steps, theta, x_win_args);

	return 0;
}
