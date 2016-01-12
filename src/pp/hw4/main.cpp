#include <cstdio>
#include <cstring>

#ifdef MPI_IMPL
	#include <mpi.h>
#endif

#include "io.hpp"
#include "apsp.hpp"
#include "../timer.hpp"

using pp::Time;
using pp::GetCurrentTime;
using pp::TimeDiff;
using pp::TimeDiffInMs;
using pp::TimeAdd;
using pp::TimeToLongInMs;
using pp::hw4::Graph;
using pp::hw4::ReadGraphFromFile;
using pp::hw4::CalcAPSP;
using pp::hw4::WriteGraphToFile;
using pp::hw4::DeleteCosts;

int main(int argc, char *argv[]) {
	Time io_start, io_time;
    const unsigned kStrMax = 128;

    // Check arguments
    if (argc < 4) {
        fprintf(stderr, "Insufficient args\n");
        fprintf(stderr, "Usage: %s input_file_name output_file_name block_size\n", argv[0]);
        return 0;
    }

    // Retrieve the arguments
    unsigned block_size;
    char in_file[kStrMax], out_file[kStrMax];

    strncpy(in_file, argv[1], kStrMax);
    strncpy(out_file, argv[2], kStrMax);
    sscanf(argv[3], "%u", &block_size);

#ifdef MPI_IMPL
    // Init MPI
    MPI_Init(&argc, &argv);

    int self_id;
    MPI_Comm_rank(MPI_COMM_WORLD, &self_id);

    // Device (GPU) Initialization
	cudaSetDevice(self_id);

    // Read the graph
    Graph *graph = NULL;
    if (self_id == 0) {
		io_start = GetCurrentTime();
        graph = ReadGraphFromFile(in_file);
		io_time = TimeDiff(io_start, GetCurrentTime());
	}
#else
	io_start = GetCurrentTime();
    Graph *graph = ReadGraphFromFile(in_file);
	io_time = TimeDiff(io_start, GetCurrentTime());
#endif

    // Record the computation time
    Time compute_start = GetCurrentTime();

    // Calculate APSP
    CalcAPSP(graph, block_size);

    // Record the end time and print the time
    printf("Whole APSP takes %ld ms.\n", TimeDiffInMs(compute_start, GetCurrentTime()));

#ifdef MPI_IMPL
    if (self_id == 0) {
		// Write to the file
		io_start = GetCurrentTime();
        WriteGraphToFile(out_file, graph);
		io_time = TimeAdd(io_time, TimeDiff(io_start, GetCurrentTime()));
		printf("File IO takes %ld ms.\n", TimeToLongInMs(io_time));

		// Release the resource
	    DeleteCosts(graph->weights);
	    delete graph;
	}

    // Finalize MPI
    MPI_Finalize();

#else
	// Write to the file
	io_start = GetCurrentTime();
    WriteGraphToFile(out_file, graph);
	io_time = TimeAdd(io_time, TimeDiff(io_start, GetCurrentTime()));
	printf("File IO takes %ld ms.\n", TimeToLongInMs(io_time));

	// Release the resource
    DeleteCosts(graph->weights);
    delete graph;
#endif

    return 0;
}
