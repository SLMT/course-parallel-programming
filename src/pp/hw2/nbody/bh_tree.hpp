#ifndef PP_HW2_NBODY_BHTREE_H_
#define PP_HW2_NBODY_BHTREE_H_

#include "nbody.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

class BHTree {

public:
	BHTree(Universe *uni, Vec2 min, Vec2 max);
	~BHTree();

	void InsertABody(int body_id);
	void PrintInDFS(); // For debugging
private:
	typedef struct {
		Vec2 center_of_mass;
		double total_mass;
		Vec2 coord_min, coord_mid, coord_max;
		int body_id; // This is only used when the node is a external node
		Node *nw, *ne, *sw, *se;
	} Node;

	Node *NewNode(int body_id, Vec2 min, Vec2 max);

	Node *root_;
	Universe *uni_;
	Vec2 root_min_, root_max_;
};

} // namespace nbody
} // namespace hw2
} // namespace pp


#endif  // PP_HW2_NBODY_BHTREE_H_
