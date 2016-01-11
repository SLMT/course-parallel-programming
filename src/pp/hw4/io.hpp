#ifndef PP_HW4_IO_H_
#define PP_HW4_IO_H_

#include <cstdio>
#include "apsp.hpp"

namespace pp {
namespace hw4 {

Graph *ReadGraphFromFile(char *file_name);
void WriteGraphToFile(char *file_name, Graph *graph);
void PrintCosts(FILE *stream, Graph *graph);

} // namespace hw4
} // namespace pp

#endif  // PP_HW4_IO_H_
