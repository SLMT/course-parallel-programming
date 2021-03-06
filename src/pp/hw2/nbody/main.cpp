#include <cstdio>
#include <cstdlib>
#include <unistd.h>

#include "nbody.hpp"
#include "../../gui.hpp"
#ifndef OMP
	#include "nbody_pthread.hpp"
#else
	#include "nbody_omp.hpp"
#endif
#include "../../timer.hpp"

using pp::hw2::nbody::Universe;
using pp::hw2::nbody::ReadFromFile;
using pp::hw2::nbody::NBodySim;
using pp::GUI;
using pp::GetCurrentTime;
using pp::TimeDiffInMs;

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
	win_len = (unsigned) strtol(argv[11], NULL, 10);

	// Record the start time
	pp::Time start = GetCurrentTime();

	// Read the input file
	pp::Time io_start = GetCurrentTime();

	Universe *uni = ReadFromFile(filename);
	uni->body_mass = mass;

	pp::Time io_end = GetCurrentTime();
	printf("IO took %ld ms.\n", TimeDiffInMs(io_start, io_end));

	// Create a GUI for demonstration
	GUI *gui = NULL;
	if (x_enabled)
		gui = new GUI(win_len, coord_len, x_min, y_min);

	// Start running
	NBodySim(uni, num_threads, delta_time, num_steps, theta, gui);

	// Print the execution time
	pp::Time end = GetCurrentTime();
	printf("The whole program took %ld ms for execution.\n", TimeDiffInMs(start, end));

	return 0;
}
