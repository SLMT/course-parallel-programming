#include <cstdio>
#include <cstring>
#include "apsp.hpp"

using pp::hw4::Graph;
using pp::hw4::ReadGraphFromFile;
using pp::hw4::CalcAPSP;
using pp::hw4::WriteGraphToFile;

int main(int argc, char const *argv[]) {
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

    // Read the graph
    Graph *graph = ReadGraphFromFile(in_file);

    // XXX: Debug
    PrintCosts(stdout, graph);

    // Calculate APSP
    CalcAPSP(graph, block_size);

    // XXX: Debug
    PrintCosts(stdout, graph);

    // Write to the file
    WriteGraphToFile(out_file, graph);

    // XXX: Debug
    //PrintCosts(stdout, graph);

    // Release the resource
    delete[] graph->weights;
    delete graph;

    return 0;
}
