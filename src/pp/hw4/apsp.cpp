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
    for (unsigned i = 0; i < nvertices * nvertices; i++) {
        graph->weights[i] = kCostInfinite;
    }

    // But, the cost from a node to itself is 0
    for (unsigned i = 0; i < nvertices; i++) {
        graph->weights[i * nvertices + i] = 0;
    }

    // Read all edges
    unsigned x, y;
    Cost w;
    for (unsigned i = 0; i < nedges; i++) {
        fscanf(in, "%u %u %u", &x, &y, &w);

        // Node numbers are start from 1
        x--;
        y--;

        // Record the weight
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
    PrintCosts(out, graph);

    // Close the file
    fclose(out);
}

void PrintCosts(FILE *stream, Graph *graph) {
    unsigned nvertices = graph->num_vertices;
    Cost cost;

    for (unsigned i = 0; i < nvertices; i++) {
        for (unsigned j = 0; j < nvertices; j++) {
            cost = graph->weights[i * nvertices + j];

            // Print the cost
            if (cost >= kCostInfinite) {
                fprintf(stream, "INF");
            } else {
                fprintf(stream, "%d", cost);
            }

            // Print a space or an end of line
            if (j < graph->num_vertices - 1) {
                fprintf(stream, " ");
            } else {
                fprintf(stream, "\n");
            }
        }
    }
}

} // namespace hw4
} // namespace pp
