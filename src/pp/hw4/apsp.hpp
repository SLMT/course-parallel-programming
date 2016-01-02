#ifndef PP_HW4_APSP_H_
#define PP_HW4_APSP_H_

#include <cstdio>
#include <climits>

namespace pp {
namespace hw4 {

typedef int Cost;

const Cost kCostInfinite = 1000000000;

typedef struct {
    unsigned num_vertices;
    Cost *weights;
} Graph;

Graph *ReadGraphFromFile(char *file_name);
void CalcAPSP(Graph *graph);
void WriteGraphToFile(char *file_name, Graph *graph);
void PrintCosts(FILE *stream, Graph *graph);

} // namespace hw4
} // namespace pp

#endif  // PP_HW4_APSP_H_
