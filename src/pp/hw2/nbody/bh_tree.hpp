#ifndef PP_HW2_NBODY_BHTREE_H_
#define PP_HW2_NBODY_BHTREE_H_

#include "nbody.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

class BHTree {

public:
	BHTree(Universe *uni);
	~BHTree();

	insertABody(int body_id);
	printInDFS(); // For debugging
private:
	typedef struct {
		Vec2 center_of_mass;
		double total_mass;
		Vec2 coord_min, coord_max;
		int body_id; // This is only used when the node is a external node
		Node *nw, *ne, *sw, *se;
	} Node;

	Node *root;
};

} // namespace nbody
} // namespace hw2
} // namespace pp


#endif  // PP_HW2_NBODY_BHTREE_H_
