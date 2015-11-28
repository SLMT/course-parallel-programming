#include "bh_tree.hpp"

#include <cstdio>
#include <vector>

#include "nbody.hpp"

using std::vector;

int main(int argc, char const *argv[]) {

	pp::hw2::nbody::Universe *uni = pp::hw2::nbody::ReadFromFile("tree_test.txt");
	uni->body_mass = 1.0;

	pp::hw2::nbody::Vec2 min = {0.0, 0.0}, max = {100.0, 100.0};
	pp::hw2::nbody::BHTree *tree = new pp::hw2::nbody::BHTree(uni, min, max);
	while (tree->IsThereMoreJobs())
		tree->DoASplittingJob();
	tree->PrintInDFS();

	return 0;
}

namespace pp {
namespace hw2 {
namespace nbody {

BHTreeNode::BHTreeNode(Vec2 min, Vec2 max) {
	coord_min_ = min;
	coord_mid_.x = (min.x + max.x) / 2;
	coord_mid_.y = (min.y + max.y) / 2;
	coord_max_ = max;
	body_id_ = kInternalNode;
	nw_ = NULL;
	ne_ = NULL;
	sw_ = NULL;
	se_ = NULL;
}

BHTreeNode::BHTreeNode(Universe *uni, int body_id, Vec2 min, Vec2 max) {
	center_of_mass_.x = uni->bodies[body_id].pos.x;
	center_of_mass_.y = uni->bodies[body_id].pos.y;
	body_count_ = 1;
	coord_min_ = min;
	coord_mid_.x = (min.x + max.x) / 2;
	coord_mid_.y = (min.y + max.y) / 2;
	coord_max_ = max;
	body_id_ = body_id;
	nw_ = NULL;
	ne_ = NULL;
	sw_ = NULL;
	se_ = NULL;
}

BHTree::BHTree(Universe *uni, Vec2 min, Vec2 max) {
	root_ = NULL;
	uni_ = uni;
	root_min_ = min;
	root_max_ = max;

	// Create the job queue
	job_queue_ = new std::list<SplittingJob>();
	pthread_mutex_init(&job_queue_mutex_, NULL);
	pthread_cond_init(&job_queue_cond_, NULL);
	job_count_ = 0;
	finish_count_ = 0;

	// Create a root node
	// XXX: Not consider the size = 0
	if (uni->num_bodies == 1) {
		// Create a external root node
		root_ = new BHTreeNode(uni, 0, min, max);
	} else {
		// Create an internal root node then split it
		root_ = new BHTreeNode(min, max);
		vector<int> *body_ids = new vector<int>();
		for (size_t i = 0; i < uni->num_bodies; i++)
			body_ids->push_back(i);
		job_count_++;
		SplitTheNode(root_, body_ids);
	}
}

BHTree::~BHTree() {
	// Traverse and free all nodes
	Delete(root_);
}

void BHTree::Delete(BHTreeNode *parent) {
	// Go deeper
	if (parent->nw_ != NULL)
		Delete(parent->nw_);
	if (parent->ne_ != NULL)
		Delete(parent->ne_);
	if (parent->sw_ != NULL)
		Delete(parent->sw_);
	if (parent->se_ != NULL)
		Delete(parent->se_);

	delete parent;
}

void BHTree::CreateRegion(BHTreeNode **region_ptr, vector<int> *bodies, Vec2 min, Vec2 max) {
	size_t size = bodies->size();
	if (size == 0)
		return;

	if (size == 1) {
		// Create a external node
		*region_ptr = new BHTreeNode(uni_, bodies->front(), min, max);
		return;
	}

	// Create an internal node
	*region_ptr = new BHTreeNode(min, max);
	InsertASplittingJob(*region_ptr, bodies);
}

void BHTree::SplitTheNode(BHTreeNode *parent, vector<int> *body_ids) {
	CelestialBody *bodies = uni_->bodies;
	Vec2 tmp = {0.0, 0.0};

	// Allocate containers for regions
	vector<int> *nw_bodies = new vector<int>();
	vector<int> *ne_bodies = new vector<int>();
	vector<int> *sw_bodies = new vector<int>();
	vector<int> *se_bodies = new vector<int>();

	// Traverse all the bodies
	Vec2 mid = parent->coord_mid_;
	for (vector<int>::iterator it = body_ids->begin(); it != body_ids->end(); ++it) {
		int id = *it;

		// Calculate the center of the mass
		tmp.x += bodies[id].pos.x;
		tmp.y += bodies[id].pos.y;

		// Put the body to a proper region
		Vec2 body_pos = (uni_->bodies[id]).pos;
		if (body_pos.x < mid.x && body_pos.y < mid.y) { // North-West
			nw_bodies->push_back(id);
		}else if (body_pos.x > mid.x && body_pos.y < mid.y) { // North-East
			ne_bodies->push_back(id);
		}else if (body_pos.x < mid.x && body_pos.y > mid.y) { // South-West
			sw_bodies->push_back(id);
		}else { // South-East
			se_bodies->push_back(id);
		}
	}

	// Record the center of mass and total mass for the node
	size_t body_count = body_ids->size();
	parent->center_of_mass_.x = tmp.x / body_count;
	parent->center_of_mass_.y = tmp.y / body_count;
	parent->body_count_ = body_count;

	// Free the input container
	delete body_ids;

	// ====================
	// == Create regions ==
	// ====================
	Vec2 min, max;

	// North-West
	min = parent->coord_min_;
	max = parent->coord_mid_;
	CreateRegion(&(parent->nw_), nw_bodies, min, max);

	// North-East
	min.x = parent->coord_mid_.x;
	min.y = parent->coord_min_.y;
	max.x = parent->coord_max_.x;
	max.y = parent->coord_mid_.y;
	CreateRegion(&(parent->ne_), ne_bodies, min, max);

	// South-West
	min.x = parent->coord_min_.x;
	min.y = parent->coord_mid_.y;
	max.x = parent->coord_mid_.x;
	max.y = parent->coord_max_.y;
	CreateRegion(&(parent->sw_), sw_bodies, min, max);

	// South-East
	min = parent->coord_mid_;
	max = parent->coord_max_;
	CreateRegion(&(parent->se_), se_bodies, min, max);

	// Critical Section
	pthread_mutex_lock(&job_queue_mutex_);

	// Finish a job, increment the finish job counter
	finish_count_++;

	// Notify threads to retrieve the pending jobs
	pthread_cond_broadcast(&job_queue_cond_);

	pthread_mutex_unlock(&job_queue_mutex_);
}

void BHTree::InsertASplittingJob(BHTreeNode *parent, vector<int> *bodies) {
	SplittingJob job = {parent, bodies};

	pthread_mutex_lock(&job_queue_mutex_);
	// Increment the job counter
	job_count_++;

	// Insert a spliting job
	job_queue_->push_back(job);

	pthread_mutex_unlock(&job_queue_mutex_);
}

void BHTree::DoASplittingJob() {
	SplittingJob job;

	// Retrieve a job
	pthread_mutex_lock(&job_queue_mutex_);

	while (job_queue_->size() == 0) {
		// If there is no more job, leave early
		if (job_count_ <= finish_count_) {
			pthread_mutex_unlock(&job_queue_mutex_);
			return;
		}
		pthread_cond_wait(&job_queue_cond_, &job_queue_mutex_);
	}

	job = job_queue_->front();
	job_queue_->pop_front();

	pthread_mutex_unlock(&job_queue_mutex_);

	// Do the job
	SplitTheNode(job.parent, job.bodies);
}

bool BHTree::IsThereMoreJobs() {
	bool result;

	pthread_mutex_lock(&job_queue_mutex_);
	result = job_count_ > finish_count_;
	pthread_mutex_unlock(&job_queue_mutex_);

	return result;
}

void BHTree::PrintInDFS() {
	PrintInDFS(root_);
}

void BHTree::PrintInDFS(BHTreeNode *node) {
	// Go deeper
	if (node->nw_ != NULL)
		PrintInDFS(node->nw_);
	if (node->ne_ != NULL)
		PrintInDFS(node->ne_);
	if (node->sw_ != NULL)
		PrintInDFS(node->sw_);
	if (node->se_ != NULL)
		PrintInDFS(node->se_);

	// Execute the action
	if (node->body_id_ != BHTreeNode::kInternalNode) {
		printf("Body Id: %d\n", node->body_id_);
	} else {
		printf("Center of mass: (%lf, %lf), count: %d\n", node->center_of_mass_.x, node->center_of_mass_.y, node->body_count_);
	}
}

} // namespace nbody
} // namespace hw2
} // namespace pp
