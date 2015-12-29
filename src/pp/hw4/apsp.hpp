#ifndef PP_HW4_APSP_H_
#define PP_HW4_APSP_H_

namespace pp {
namespace hw4 {

typedef struct {
    unsigned num_vertices;
    unsigned *weights;
} Graph;

Graph *ReadGraphFromFile(char *file_name);

} // namespace hw4
} // namespace pp

#endif  // PP_HW4_APSP_H_
