#ifndef PP_HW4_APSP_H_
#define PP_HW4_APSP_H_

namespace pp {
namespace hw4 {

typedef int Cost;

const Cost kCostInfinite = 1000000000;

typedef struct {
    unsigned num_vertices;
    Cost *weights;
} Graph;

Cost *NewCosts(unsigned num_costs);
void DeleteCosts(Cost *costs);
void CalcAPSP(Graph *graph, unsigned block_size);

} // namespace hw4
} // namespace pp

#endif  // PP_HW4_APSP_H_
