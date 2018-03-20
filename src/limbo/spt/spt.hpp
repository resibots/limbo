//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_SPT_SPT_HPP
#define LIMBO_SPT_SPT_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <limbo/tools/random_generator.hpp>
#include <memory>
#include <stack>
#include <vector>

namespace limbo {
    namespace spt {
        /// A implementation of Spatial Trees as described in
        /// McFee, B. and Lanckriet, G.R., 2011. Large-scale music similarity search with spatial trees. In ISMIR (pp. 55-60).
        struct SpatialTreeNode {
        public:
            SpatialTreeNode() : _parent(nullptr), _left(nullptr), _right(nullptr), _is_left(false), _depth(0) {}

            SpatialTreeNode(const std::vector<Eigen::VectorXd>& points, const std::vector<Eigen::VectorXd>& observations)
                : _parent(nullptr), _left(nullptr), _right(nullptr), _points(points), _observations(observations), _is_left(false), _depth(0) {}

            const std::vector<Eigen::VectorXd>& points() const { return _points; }
            const std::vector<Eigen::VectorXd>& observations() const { return _observations; }

            double split_median() { return _split_median; }
            void set_split_median(double sp_med) { _split_median = sp_med; }

            double split_median_left() const { return _split_median_left; }
            void set_split_median_left(double sp_med) { _split_median_left = sp_med; }

            double split_median_right() const { return _split_median_right; }
            void set_split_median_right(double sp_med) { _split_median_right = sp_med; }

            const Eigen::VectorXd& split_dir() const { return _split_dir; }
            void set_split_dir(const Eigen::VectorXd& sp_dir) { _split_dir = sp_dir; }

            const Eigen::VectorXd& split_vector() const { return _split_vector; }
            void set_split_vector(const Eigen::VectorXd& sp_vec) { _split_vector = sp_vec; }

            const std::shared_ptr<SpatialTreeNode>& left() const { return _left; }
            void set_left(const std::shared_ptr<SpatialTreeNode>& left) { _left = left; }

            const std::shared_ptr<SpatialTreeNode>& right() const { return _right; }
            void set_right(const std::shared_ptr<SpatialTreeNode>& right) { _right = right; }

            const std::shared_ptr<SpatialTreeNode>& parent() const { return _parent; }
            void set_parent(const std::shared_ptr<SpatialTreeNode>& parent) { _parent = parent; }

            const Eigen::VectorXd& max() const { return _max; }
            void set_max(const Eigen::VectorXd& max) { _max = max; }

            const Eigen::VectorXd& min() const { return _min; }
            void set_min(const Eigen::VectorXd& min) { _min = min; }

            bool is_left() const { return _is_left; }
            void make_left() { _is_left = true; }

            size_t depth() const { return _depth; }
            void set_depth(size_t d) { _depth = d; }

            const std::vector<Eigen::VectorXd>& boundary_points() const { return _boundary_points; }
            void set_boundary_points(const std::vector<Eigen::VectorXd>& boundary_points) { _boundary_points = boundary_points; }

        protected:
            std::shared_ptr<SpatialTreeNode> _parent, _left, _right;
            std::vector<Eigen::VectorXd> _points, _observations, _boundary_points;
            Eigen::VectorXd _split_dir, _split_vector;
            Eigen::VectorXd _max, _min;
            double _split_median, _split_median_left, _split_median_right;
            bool _is_left;
            size_t _depth;
        };

        inline Eigen::MatrixXd sample_covariance(const std::vector<Eigen::VectorXd>& points)
        {
            assert(points.size());

            // Get the sample means
            Eigen::VectorXd means = Eigen::VectorXd::Zero(points[0].size());

            for (size_t i = 0; i < points.size(); i++) {
                means.array() += points[i].array();
            }

            means = means.array() / double(points.size());

            // Calculate the sample covariance matrix
            Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(points[0].size(), points[0].size());
            for (size_t i = 0; i < points.size(); i++) {
                cov = cov + points[i] * points[i].transpose();
            }

            cov = (cov.array() - (double(points.size()) * means * means.transpose()).array()) / (double(points.size()) - 1.0);

            return cov;
        }

        inline Eigen::VectorXd get_split_dir(const std::vector<Eigen::VectorXd>& points, int depth)
        {
            assert(points.size());

            // // KD-tree
            // Eigen::VectorXd split_dir = Eigen::VectorXd::Zero(points[0].size());
            // int in = depth % points[0].size();
            // split_dir(in) = 1.0;
            //
            // return split_dir;

            // // In 1-D we always split in the same direction (there is afterall only one direction)
            // if (points[0].size() == 1)
            //     return limbo::tools::make_vector(1);

            Eigen::MatrixXd cov = sample_covariance(points);

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(cov);
            Eigen::MatrixXd eig = solver.eigenvectors();
            Eigen::VectorXd eig_v = solver.eigenvalues();
            // std::cout << eig.col(0)[0].real() << " " << eig.col(0)[1].real() << " -> " << std::sqrt((eig.col(0)[0].real() * eig.col(0)[0].real()) + (eig.col(0)[1].real() * eig.col(0)[1].real())) << std::endl;
            // std::cout << eig.col(1)[0].real() << " " << eig.col(1)[1].real() << " -> " << std::sqrt((eig.col(1)[0].real() * eig.col(1)[0].real()) + (eig.col(1)[1].real() * eig.col(1)[1].real())) << std::endl;
            // std::cout << std::endl;

            Eigen::VectorXd best_eig;
            double best = -std::numeric_limits<double>::max();
            for (int i = 0; i < eig.cols(); i++) {
                Eigen::VectorXd v(points[0].size());
                for (int j = 0; j < v.size(); j++) {
                    v(j) = eig.col(i)[j]; //.real();
                }
                // if (std::abs(v.norm() - 1.0) > 1e-8)
                //     continue;
                double val = eig_v(i); //.real(); //v.transpose() * cov * v;
                if (val > best) {
                    best = val;
                    best_eig = v;
                }
            }

            return best_eig;
        }

        inline std::vector<double> transform_points(const std::vector<Eigen::VectorXd>& points, const Eigen::VectorXd& split_dir)
        {
            std::vector<double> p(points.size(), 0.0);

            for (size_t i = 0; i < points.size(); i++) {
                p[i] = split_dir.dot(points[i]);
            }

            return p;
        }

        inline double get_median(const std::vector<double>& points)
        {
            assert(points.size());

            int size = points.size();
            std::vector<double> p_i = points;

            std::sort(p_i.begin(), p_i.end());

            double median = (size % 2) ? p_i[size / 2] : (p_i[size / 2 - 1] + p_i[size / 2]) / 2.0;

            return median;
        }

        inline double get_quantile(const std::vector<double>& points, double perc = 0.5)
        {
            assert(points.size());

            int size = points.size();
            std::vector<double> p_i = points;

            std::sort(p_i.begin(), p_i.end());

            // OLD IMPLEMENTATION -- no linear interpolation
            // double index = perc * size;
            // int i = std::floor(index);
            // // std::cout << i << " " << index << std::endl;
            // if (i == index && i > 0) {
            //     return (p_i[i - 1] + p_i[i]) / 2.0;
            // }
            //
            // return p_i[i];

            double pp = perc * (size - 1.0);
            pp = std::round(pp * 1000.0) / 1000.0;
            int ind_below = std::floor(pp);
            int ind_above = std::ceil(pp);

            if (ind_below == ind_above)
                return p_i[ind_below];

            return p_i[ind_below] * (double(ind_above) - pp) + p_i[ind_above] * (pp - double(ind_below));
        }

        // Do not call it on leaves
        inline bool in_bounds(const std::shared_ptr<SpatialTreeNode>& node, const Eigen::VectorXd& point, double eps = 1e-6)
        {
            if (!node)
                return true;

            auto p = node->parent();
            auto c = node;
            while (p) {
                Eigen::VectorXd sp = p->split_dir();
                double m = p->split_median();
                double d = sp.dot(point);

                if (!(d <= (m + eps) && c->is_left()) && !(d >= (m - eps) && !c->is_left())) {
                    return false;
                }

                c = p;
                p = p->parent();
            }

            return true;
        }

        // Do not call it on leaves
        inline void sample_boundary_points(const std::shared_ptr<SpatialTreeNode>& node, int N = 1000)
        {
            Eigen::VectorXd min_p = node->min();
            Eigen::VectorXd max_p = node->max();

            Eigen::VectorXd sp = node->split_dir();
            double m = node->split_median();

            std::vector<Eigen::VectorXd> points;
            for (int i = 0; i < N; i++) {
                Eigen::VectorXd point;
                bool b;
                do {
                    b = true;
                    int index = min_p.size() - 1;
                    while (index >= 0 && std::abs(sp(index)) < 1e-6)
                        index--;
                    assert(index >= 0);
                    // sample points on the hyper-plane
                    point = limbo::tools::random_vector(min_p.size()).array() * (max_p - min_p).array() + min_p.array();
                    Eigen::VectorXd sp_new = sp;
                    sp_new(index) = 0.0;
                    point(index) = (m - sp_new.dot(point)) / double(sp(index));

                    if (point(index) > max_p(index))
                        b = false;
                    if (point(index) < min_p(index))
                        b = false;
                } while (!b || !in_bounds(node, point));

                points.push_back(point);
            }

            node->set_boundary_points(points);
        }

        inline std::vector<Eigen::VectorXd> get_shared_boundaries(const std::shared_ptr<SpatialTreeNode>& node1, const std::shared_ptr<SpatialTreeNode>& node2)
        {
            auto p1 = node1->parent();
            auto p2 = node2->parent();

            while (p1 && p2) {
                if (p1 == p2)
                    break;
                p1 = p1->parent();
                p2 = p2->parent();
            }

            if (p1 != p2)
                return std::vector<Eigen::VectorXd>();

            // common ancestor
            auto p = p1;

            std::vector<Eigen::VectorXd> res, b;

            b = p->boundary_points();

            for (auto bp : b) {
                if (in_bounds(node1, bp) && in_bounds(node2, bp))
                    res.push_back(bp);
            }

            return res;
        }

        inline std::shared_ptr<SpatialTreeNode> make_spt(const std::vector<Eigen::VectorXd>& points, const std::vector<Eigen::VectorXd>& observations, int max_depth = 2, double tau = 0.1, int depth = 0)
        {
            auto spt_node = std::make_shared<SpatialTreeNode>(points, observations);
            spt_node->set_depth(depth);

            if (max_depth == 0) {
                return spt_node;
            }

            // calculate split direction
            Eigen::VectorXd split_dir = get_split_dir(points, depth);

            // calculate median
            std::vector<double> transformed_points = transform_points(points, split_dir);
            double split_median = get_median(transformed_points);
            double split_median_left = get_quantile(transformed_points, 0.5 + tau);
            double split_median_right = get_quantile(transformed_points, 0.5 - tau);

            spt_node->set_split_dir(split_dir);
            spt_node->set_split_median(split_median);
            spt_node->set_split_median_left(split_median_left);
            spt_node->set_split_median_right(split_median_right);

            // get new points
            double min = std::numeric_limits<double>::max();
            int min_i = -1;
            std::vector<Eigen::VectorXd> left_points, right_points, left_obs, right_obs;

            Eigen::VectorXd min_left = Eigen::VectorXd::Constant(points[0].size(), std::numeric_limits<double>::max());
            Eigen::VectorXd max_left = Eigen::VectorXd::Constant(points[0].size(), -std::numeric_limits<double>::max());
            Eigen::VectorXd min_right = Eigen::VectorXd::Constant(points[0].size(), std::numeric_limits<double>::max());
            Eigen::VectorXd max_right = Eigen::VectorXd::Constant(points[0].size(), -std::numeric_limits<double>::max());

            Eigen::VectorXd min_p = Eigen::VectorXd::Constant(points[0].size(), std::numeric_limits<double>::max());
            Eigen::VectorXd max_p = Eigen::VectorXd::Constant(points[0].size(), -std::numeric_limits<double>::max());

            // double min_left2 = std::numeric_limits<double>::max();
            // double min_right2 = std::numeric_limits<double>::max();
            // int min_i_left = -1;
            // int min_i_right = -1;

            for (size_t i = 0; i < points.size(); i++) {
                // this is needed for the boundaries computation
                for (int j = 0; j < points[0].size(); j++) {
                    if (points[i](j) < min_p(j)) {
                        min_p(j) = points[i](j);
                    }
                    if (points[i](j) > max_p(j)) {
                        max_p(j) = points[i](j);
                    }
                }

                if (transformed_points[i] <= split_median_left) {
                    left_points.push_back(points[i]);
                    left_obs.push_back(observations[i]);
                    // // this is needed for the boundaries computation
                    // for (int j = 0; j < points[0].size(); j++) {
                    //     if (points[i](j) < min_left(j)) {
                    //         min_left(j) = points[i](j);
                    //     }
                    //     if (points[i](j) > max_left(j)) {
                    //         max_left(j) = points[i](j);
                    //     }
                    // }

                    // double dist = std::abs(transformed_points[i] - split_median);
                    // if (dist < min_left2) {
                    //     min_left2 = dist;
                    //     min_i_left = i;
                    // }
                }
                if (transformed_points[i] > split_median_right) {
                    right_points.push_back(points[i]);
                    right_obs.push_back(observations[i]);
                    // // this is needed for the boundaries computation
                    // for (int j = 0; j < points[0].size(); j++) {
                    //     if (points[i](j) < min_right(j)) {
                    //         min_right(j) = points[i](j);
                    //     }
                    //     if (points[i](j) > max_right(j)) {
                    //         max_right(j) = points[i](j);
                    //     }
                    // }

                    // double dist = std::abs(transformed_points[i] - split_median);
                    // if (dist < min_right2) {
                    //     min_right2 = dist;
                    //     min_i_right = i;
                    // }
                }

                double dist = std::abs(transformed_points[i] - split_median);
                if (dist < min) {
                    min = dist;
                    min_i = i;
                }
            }

            // Get split point (closer to median)
            spt_node->set_split_vector(points[min_i]);
            spt_node->set_max(max_p);
            spt_node->set_min(min_p);
            // std::cout << depth << ": " << min_p.transpose() << " -> " << max_p.transpose() << std::endl;

            // TO-DO: Check if we should put it back
            // sample_boundary_points(spt_node); //, max_depth * max_depth * 7);

            // Make left node
            auto left_node = make_spt(left_points, left_obs, max_depth - 1, tau, depth + 1);
            left_node->set_parent(spt_node);
            left_node->make_left();
            spt_node->set_left(left_node);

            // left_node->set_max(max_left);
            // left_node->set_min(min_left);
            // left_node->set_split_vector(points[min_i_left]);

            // Make right node
            auto right_node = make_spt(right_points, right_obs, max_depth - 1, tau, depth + 1);
            right_node->set_parent(spt_node);
            spt_node->set_right(right_node);

            // right_node->set_max(max_right);
            // right_node->set_min(min_right);
            // right_node->set_split_vector(points[min_i_right]);

            return spt_node;
        }

        inline std::vector<std::shared_ptr<SpatialTreeNode>> get_leaves(const std::shared_ptr<SpatialTreeNode>& spt_node)
        {
            std::vector<std::shared_ptr<SpatialTreeNode>> leaves;

            std::stack<std::shared_ptr<SpatialTreeNode>> S;
            S.push(spt_node);

            while (!S.empty()) {
                auto n = S.top();
                S.pop();

                auto l = n->left();
                auto r = n->right();

                if (l->left() == nullptr || l->right() == nullptr) {
                    leaves.push_back(l);
                }
                else
                    S.push(l);

                if (r->left() == nullptr || r->right() == nullptr) {
                    leaves.push_back(r);
                }
                else
                    S.push(r);
            }

            return leaves;
        }
    } // namespace spt
} // namespace limbo

#endif