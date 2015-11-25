#include "bh_tree.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

BHTree::BHTree(Universe *uni, Vec2 min, Vec2 max) {
	root_ = NULL;
	uni_ = uni;
	root_min_ = min;
	root_max_ = max;
}

BHTree::~BHTree() {
	// TODO: Traverse and free all nodes
}

Node *BHTree::NewNode(int body_id, Vec2 min, Vec2 max) {
	Node *node = new Node();

	node->center_of_mass = {uni_[body_id].pos.x, uni_[body_id].pos.y};
	node->total_mass = uni_->body_mass;
	node->coord_min = min;
	node->coord_mid.x = (min.x + max.x) / 2;
	node->coord_mid.y = (min.y + max.y) / 2;
	node->coord_max = max;
	node->body_id = body_id;
	node->nw = NULL;
	node->ne = NULL;
	node->sw = NULL;
	node->se = NULL;

	return node;
}

void BHTree::InsertABody(int body_id) {
	Vec2 body_pos = (uni_->bodies[body_id]).pos;
	Vec2 new_min, new_max;

	// If there is no root
	if (root_ == NULL) {
		root_ = new NewNode(body_id, root_min_, root_max_);
		return;
	}

	// If there is a root, find the region it should be in
	Node *parent = root_;
	Node **target;
	while () {
		// TODO: Check if the node is a external node
		// TODO: Write a method to put a node at a correct position

		Vec2 min = parent->coord_min;
		Vec2 mid = parent->coord_mid;
		Vec2 max = parent->coord_max;

		if (body_pos.x < mid.x && body_pos.y < mid.y) { // North-West
			if (parent->nw == NULL) {
				new_min = min;
				new_max = mid;
			}
			target = &(parent->nw);
		} else if (body_pos.x > mid.x && body_pos.y < mid.y) { // North-East
			if (parent->ne == NULL) {
				new_min.x = mid.x;
				new_min.y = min.y;
				new_max.x = max.x;
				new_max.y = mid.y;
			}
			target = &(parent->ne);
		} else if (body_pos.x < mid.x && body_pos.y > mid.y) { // South-West
			if (parent->sw == NULL) {
				new_min.x = min.x;
				new_min.y = mid.y;
				new_max.x = mid.x;
				new_max.y = max.y;
			}
			target = &(parent->sw);
		} else { // South-East
			if (parent->se == NULL) {
				new_min = mid;
				new_max = max;
			}
			target = &(parent->se);
		}

		// If there is no such child yet, create one then update root
		if (*target == NULL) {
			*target = new NewNode(body_id, new_min, new_max);
			// TODO: Update the mass and center of the parent

			return;
		}

		// TODO: Update the mass and center of the parent by child
		// TODO: Go down next level
	}
}

void BHTree::PrintInDFS() {
	// TODO: Traverse and print all nodes
}


} // namespace nbody
} // namespace hw2
} // namespace pp
