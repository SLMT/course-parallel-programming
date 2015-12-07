#ifndef PP_HW2_NBODY_BHTREE_H_
#define PP_HW2_NBODY_BHTREE_H_

#include <list>
#include <vector>
#include <pthread.h>

#include "nbody.hpp"
#include "../../gui.hpp"

using std::vector;
using std::list;

namespace pp {
namespace hw2 {
namespace nbody {

class BHTreeNode {
	friend class BHTree;

public:
	BHTreeNode(Vec2 min, Vec2 max);
	BHTreeNode(Universe *uni, int body_id, Vec2 min, Vec2 max);

private:
	static const int kInternalNode = -1;

	Vec2 center_of_mass_;
	int body_count_;
	Vec2 coord_min_, coord_mid_, coord_max_;
	int body_id_; // This is only used when the node is a external node
	BHTreeNode *nw_, *ne_, *sw_, *se_;
};

class BHTree {

public:
	BHTree(Universe *uni);
	~BHTree();

	// For multi-threading splitting
	void DoASplittingJob();
	bool IsThereMoreJobs();

	// Calculate force
	Vec2 CalculateTotalForce(int source_id, double theta);
	Vec2 CalculateTotalForce(int source_id, double theta, BHTreeNode *node);

	// Free all resource
	void Delete(BHTreeNode *parent);

	// For drawing
	void DrawRegions(GUI *gui);
	void DrawRegions(GUI *gui, BHTreeNode *node);

	// For debugging
	void PrintInDFS();
	void PrintInDFS(BHTreeNode *node);

private:

	// For spliting
	typedef struct {
		BHTreeNode *parent;
		vector<int> *bodies;
	} SplittingJob;
	list<SplittingJob> *job_queue_;
	pthread_mutex_t job_queue_mutex_;
	pthread_cond_t job_queue_cond_;
	size_t job_count_, finish_count_;

	void InsertASplittingJob(BHTreeNode *parent, vector<int> *bodies);
	void SplitTheNode(BHTreeNode *parent, vector<int> *body_ids);
	void CreateRegion(BHTreeNode **region_ptr, vector<int> *bodies, Vec2 min, Vec2 max);

	// Properties
	BHTreeNode *root_;
	Universe *uni_;
	Vec2 root_min_, root_max_;
};

} // namespace nbody
} // namespace hw2
} // namespace pp


#endif  // PP_HW2_NBODY_BHTREE_H_
