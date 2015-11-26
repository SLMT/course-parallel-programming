#ifndef PP_HW2_NBODY_BHTREE_H_
#define PP_HW2_NBODY_BHTREE_H_

#include <list>

#include "nbody.hpp"

namespace pp {
namespace hw2 {
namespace nbody {

class BHTree {

public:
	BHTree(Universe *uni, Vec2 min, Vec2 max);
	~BHTree();

	// For multi-threading
	void DoASplittingJob();

	// For debugging
	void PrintInDFS();

	// TODO: Add a API for check job status (more jobs)

private:
	// Node structure
	typedef struct {
		Vec2 center_of_mass;
		int body_count;
		Vec2 coord_min, coord_mid, coord_max;
		int body_id; // This is only used when the node is a external node
		Node *nw, *ne, *sw, *se;
	} Node;

	Node *NewNode(int body_id, Vec2 min, Vec2 max);

	// For spliting
	typedef struct {
		Node *parent;
		vector<int> *bodies;
	} SplittingJob;
	std::list<SplittingJob> *job_queue_;
	pthread_mutex_t job_queue_mutex_;
	pthread_cond_t job_queue_cond_;

	void InsertASplittingJob(Node *parent, vector<int> *bodies);
	void SplitTheNode(Node *parent, vector<int> *body_ids);
	void CreateRegion(Node **region_ptr, vector<int> *bodies, Vec2 min, Vec2 max);

	// Properties
	Node *root_;
	Universe *uni_;
	Vec2 root_min_, root_max_;
};

} // namespace nbody
} // namespace hw2
} // namespace pp


#endif  // PP_HW2_NBODY_BHTREE_H_
