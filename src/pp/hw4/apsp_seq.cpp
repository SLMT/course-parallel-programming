#include "apsp.hpp"

namespace pp {
namespace hw4 {

void CalcAPSP(Graph *graph) {
	unsigned nvertices = graph->num_vertices;
	Cost *weights = graph->weights;

	Cost new_cost;
	for (unsigned k = 0; k < nvertices; k++) {
		for (unsigned i = 0; i < nvertices; i++) {
			for (unsigned j = 0; j < nvertices; j++) {
				new_cost = weights[i * nvertices + k] + weights[k * nvertices + j];

				if (weights[i * nvertices + j] > new_cost) {
					weights[i * nvertices + j] = new_cost;
				}
			}
		}
	}
}

}
}
