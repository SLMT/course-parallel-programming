#ifndef PP_HW4_BLOCKCALCULATION_H_
#define PP_HW4_BLOCKCALCULATION_H_

#include "apsp.hpp"

namespace pp {
namespace hw4 {

void CalcBlocks(Cost *costs, unsigned num_nodes, unsigned block_size, unsigned round_idx, unsigned block_x_start, unsigned block_y_start, unsigned block_x_len, unsigned block_y_len);

} // namespace hw4
} // namespace pp

#endif  // PP_HW4_BLOCKCALCULATION_H_
