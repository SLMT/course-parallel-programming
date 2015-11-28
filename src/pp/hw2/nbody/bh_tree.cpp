#include "bh_tree.hpp"

#include <cstdio>
#include <vector>

using std::vector;

namespace pp {
namespace hw2 {
namespace nbody {

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
		root_ = NewNode(0, min, max);
	} else {
		// Create an internal root node then split it
		root_ = NewNode(INTERNAL_NODE, min, max);
		vector<int> body_ids = new vector<int>();
		for (size_t i = 0; i < uni->num_bodies; i++)
			body_ids.push_back(i);
		SplitTheNode(root_, body_ids);
	}
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

void BHTree::CreateRegion(Node **region_ptr, vector<int> *bodies, Vec2 min, Vec2 max) {
	size_t size = bodies.size();
	if (size == 0)
		return;

	if (size == 1) {
		// Create a external node
		*region_ptr = NewNode(bodies->front(), min, max);
		return;
	}

	// Create an internal node
	*region_ptr = NewNode(INTERNAL_NODE, min, max);
	InsertASplitingJob(*region_ptr, bodies);
}

void BHTree::SplitTheNode(Node *parent, vector<int> *body_ids) {
	CelestialBody *bodies = uni_->bodies;
	Vec2 tmp;

	// Allocate containers for regions
	vector<int> *nw_bodies = new vector<int>();
	vector<int> *ne_bodies = new vector<int>();
	vector<int> *sw_bodies = new vector<int>();
	vector<int> *se_bodies = new vector<int>();

	// Traverse all the bodies
	Vec2 mid = parent->coord_mid;
	for (vector<int>::iterator it = body_ids->begin(); it != body_ids->end(); ++it) {
		int id = *it;

		// Calculate the center of the mass
		tmp.x += bodies[id].pos.x;
		tmp.y += bodies[id].pos.y;

		// Put the body to a proper region
		Vec2 body_pos = (uni_->bodies[body_id]).pos;
		if (body_pos.x < mid.x && body_pos.y < mid.y) // North-West
			nw_bodies->push_back();
		else if (body_pos.x > mid.x && body_pos.y < mid.y) // North-East
			ne_bodies->push_back();
		else if (body_pos.x < mid.x && body_pos.y > mid.y) // South-West
			sw_bodies->push_back();
		else // South-East
			se_bodies->push_back();
	}

	// Record the center of mass and total mass for the node
	size_t body_count = body_ids->size();
	parent->center_of_mass.x = tmp.x / body_count;
	parent->center_of_mass.y = tmp.y / body_count;
	parent->body_count = body_count;

	// Free the input container
	delete body_ids;

	// == Create regions ==
	Vec2 min, max;

	// North-West
	min = parent->coord_min;
	max = parent->coord_max;
	CreateRegion(&(parent->nw), nw_bodies, min, max);

	// North-East
	min.x = parent->coord_mid.x;
	min.y = parent->coord_min.y;
	max.x = parent->coord_max.x;
	max.y = parent->coord_mid.y;
	CreateRegion(&(parent->ne), ne_bodies, min, max);

	// South-West
	min.x = parent->coord_min.x;
	min.y = parent->coord_mid.y;
	max.x = parent->coord_mid.x;
	max.y = parent->coord_max.y;
	CreateRegion(&(parent->sw), sw_bodies, min, max);

	// South-East
	min = parent->coord_mid;
	max = parent->coord_max;
	CreateRegion(&(parent->se), se_bodies, min, max);

	// Critical Section
	pthread_mutex_lock(&job_queue_mutex_);

	// Finish a job, increment the finish job counter
	finish_count_++;

	// Notify threads to retrieve the pending jobs
	pthread_cond_broadcast(&job_queue_cond_);

	pthread_mutex_unlock(&job_queue_mutex_);
}

void BHTree::InsertASplitingJob(Node *parent, vector<int> *bodies) {
	SplittingJob job = {parent, bodies};

	pthread_mutex_lock(&job_queue_mutex_);
	// Increment the job counter
	job_count_++;

	// Insert a spliting job
	job_queue_->push_back(job);

	pthread_mutex_unlock(&job_queue_mutex_);
}

void BHTree::DoASplitingJob() {
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

	job = job_queue_->pop_front();

	pthread_mutex_unlock(&job_queue_mutex_);

	// Do the job
	SplitTheNode(job.parent, job.bodies);
}

void BHTree::IsThereMoreJobs() {
	bool result;

	pthread_mutex_lock(&job_queue_mutex_);
	result = job_count_ > finish_count_;
	pthread_mutex_unlock(&job_queue_mutex_);

	return result;
}

void BHTree::PrintInDFS() {
	DFS(root_, &PrintNode);
}

void BHTree::DFS(Node *node, void (*action)(Node *node)) {
	// Go deeper
	if (node->nw != NULL)
		DFS(node->nw, action);
	if (node->ne != NULL)
		DFS(node->ne, action);
	if (node->sw != NULL)
		DFS(node->sw, action);
	if (node->se != NULL)
		DFS(node->se, action);

	// Execute the action
	if (node->body_id != INTERNAL_NODE)
		action(node);
}

void BHTree::PrintNode(Node *node) {
	printf("Body Id: %d\n", node->body_id);
}

} // namespace nbody
} // namespace hw2
} // namespace pp
