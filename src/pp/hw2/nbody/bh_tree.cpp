#include "bh_tree.hpp"

#include <cfloat>
#include <cstdio>
#include <cmath>
#include <vector>

#include "nbody.hpp"
#include "../../gui.hpp"

using std::vector;

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

BHTree::BHTree(Universe *uni) {
	root_ = NULL;
	uni_ = uni;

	// Calculate the max and the min
	root_min_.x = DBL_MAX;
	root_min_.y = DBL_MAX;
	root_max_.x = DBL_MIN;
	root_max_.y = DBL_MIN;
	CelestialBody *bodies = uni->bodies;
	for (size_t i = 0; i < uni->num_bodies; i++) {
		if (root_min_.x > bodies[i].pos.x)
			root_min_.x = bodies[i].pos.x;
		if (root_min_.y > bodies[i].pos.y)
			root_min_.y = bodies[i].pos.y;
		if (root_max_.x < bodies[i].pos.x)
			root_max_.x = bodies[i].pos.x;
		if (root_max_.y < bodies[i].pos.y)
			root_max_.y = bodies[i].pos.y;
	}

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
		root_ = new BHTreeNode(uni, 0, root_min_, root_max_);
	} else {
		// Create an internal root node then split it
		root_ = new BHTreeNode(root_min_, root_max_);
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
		int pos = 0;
		Vec2 body_pos = (uni_->bodies[id]).pos;
		if (body_pos.x <= mid.x && body_pos.y <= mid.y) { // North-West
			nw_bodies->push_back(id);
			pos = 1;
		}else if (body_pos.x >= mid.x && body_pos.y <= mid.y) { // North-East
			ne_bodies->push_back(id);
			pos = 2;
		}else if (body_pos.x <= mid.x && body_pos.y >= mid.y) { // South-West
			sw_bodies->push_back(id);
			pos = 3;
		}else { // South-East
			se_bodies->push_back(id);
			pos = 4;
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

Vec2 BHTree::CalculateTotalForce(int source_id, double theta) {
	return CalculateTotalForce(source_id, theta, root_);
}

Vec2 BHTree::CalculateTotalForce(int source_id, double theta, BHTreeNode *node) {
	Vec2 total_force = {0.0, 0.0}, dis;
	double target_mass, dis_total;
	CelestialBody *bodies = uni_->bodies;

	// If this node is the source node
	if (node->body_id_ == source_id) {
		return total_force;
	}
	// For an ineternal node
	else if (node->body_id_ == BHTreeNode::kInternalNode) {
		dis.x = node->center_of_mass_.x - bodies[source_id].pos.x;
		dis.y = node->center_of_mass_.y - bodies[source_id].pos.y;
		dis_total = sqrt(dis.x * dis.x + dis.y * dis.y);
		double region_width = node->coord_max_.x - node->coord_min_.x;

		// Check if we need to go deeper
		if (region_width / dis_total > theta) {
			// Go deeper
			if (node->nw_ != NULL)
				total_force = Add(CalculateTotalForce(source_id, theta, node->nw_), total_force);
			if (node->ne_ != NULL)
				total_force = Add(CalculateTotalForce(source_id, theta, node->ne_), total_force);
			if (node->sw_ != NULL)
				total_force = Add(CalculateTotalForce(source_id, theta, node->sw_), total_force);
			if (node->se_ != NULL)
				total_force = Add(CalculateTotalForce(source_id, theta, node->se_), total_force);
			return total_force;
		}

		target_mass = uni_->body_mass * node->body_count_;
	} else { // For an external node
		dis.x = bodies[node->body_id_].pos.x - bodies[source_id].pos.x;
		dis.y = bodies[node->body_id_].pos.y - bodies[source_id].pos.y;
		target_mass = uni_->body_mass;
	}

	// Apply the formula
	return GravitationFormula(uni_->body_mass, target_mass, dis);
}

void BHTree::DrawRegions(GUI *gui) {
	// Draw boundaries
	gui->DrawALine(root_min_.x, root_min_.y, root_min_.x, root_max_.y);
	gui->DrawALine(root_min_.x, root_max_.y, root_max_.x, root_max_.y);
	gui->DrawALine(root_max_.x, root_max_.y, root_max_.x, root_min_.y);
	gui->DrawALine(root_max_.x, root_min_.y, root_min_.x, root_min_.y);

	DrawRegions(gui, root_);
}

void BHTree::DrawRegions(GUI *gui, BHTreeNode *node) {
	// Draw the middle line of the current region
	if (node->body_id_ == BHTreeNode::kInternalNode) {
		Vec2 min = node->coord_min_;
		Vec2 mid = node->coord_mid_;
		Vec2 max = node->coord_max_;
		gui->DrawALine(mid.x, min.y, mid.x, max.y);
		gui->DrawALine(min.x, mid.y, max.x, mid.y);
	}

	// Go deeper
	if (node->nw_ != NULL)
		DrawRegions(gui, node->nw_);
	if (node->ne_ != NULL)
		DrawRegions(gui, node->ne_);
	if (node->sw_ != NULL)
		DrawRegions(gui, node->sw_);
	if (node->se_ != NULL)
		DrawRegions(gui, node->se_);
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
