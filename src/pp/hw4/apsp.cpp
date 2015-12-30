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
    graph->weights = new Cost[nvertices * nvertices];

    // Fill in all space with "infinite"
    for (unsigned i = 0; i < nvertices * nvertices; i++)
        graph->weights[i] = kCostInfinite;

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

void WriteGraphToFile(char *file_name, Graph *graph) {
    // Open the file
    FILE *out = fopen(file_name, "w");

    // Write shortest distance
    for (unsigned i = 0; i < graph->num_vertices; i++) {
        for (unsigned j = 0; j < graph->num_vertices; j++) {
            fprintf(out, "%d", graph->weights[i * nvertices + j]);

            if (j < graph->num_vertices - 1) {
                fprintf(out, " ");
            } else {
                fprintf(out, "\n");
            }
        }
    }

    // Close the file
    fclose(out);
}

} // namespace hw4
} // namespace pp
