#include "apsp.hpp"

#include <cstdio>

namespace pp {
namespace hw4 {

Graph *ReadGraphFromFile(char *file_name) {

    // Open the file
    FILE *in = fopen(file_name, "r");

    // Get the # of vertices and edges
    unsigned nvertices, nedges;
    fscanf(in, "%u %u", &nvertices, &nedges);

    // Allocate a space for the matrix
    Graph *graph = new Graph();
    graph->num_vertices = nvertices;
    graph->weights = new unsigned[nvertices * nvertices];

    // Read all edges
    unsigned x, y;
    Cost w;
    for (unsigned i = 0; i < nedges; i++) {
        fscanf("%u %u %u", &x, &y, &w);
        graph->weights[x * nvertices + y] = w;
    }

    // Close the file
    fclose(in);

    return graph;
}

} // namespace hw4
} // namespace pp
