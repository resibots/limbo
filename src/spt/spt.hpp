#ifndef SPT_SPT_HPP
#define SPT_SPT_HPP

#include <Eigen/Core>
#include <limbo/opt.hpp>
#include <algorithm>
#include <vector>
#include <stack>
#include <memory>

namespace spt {

    struct SPTNode {
    public:
        SPTNode() : _parent(nullptr), _left(nullptr), _right(nullptr) {}

        SPTNode(const std::vector<Eigen::VectorXd>& points)
            : _parent(nullptr), _left(nullptr), _right(nullptr), _points(points) {}

        std::vector<Eigen::VectorXd> points() { return _points; }

        double split_median() { return _split_median; }
        void set_split_median(double sp_med) { _split_median = sp_med; }

        Eigen::VectorXd split_dir() { return _split_dir; }
        void set_split_dir(const Eigen::VectorXd& sp_dir) { _split_dir = sp_dir; }

        Eigen::VectorXd split_vector() { return _split_vector; }
        void set_split_vector(const Eigen::VectorXd& sp_vec) { _split_vector = sp_vec; }

        std::shared_ptr<SPTNode> left() { return _left; }
        void set_left(std::shared_ptr<SPTNode> left) { _left = left; }

        std::shared_ptr<SPTNode> right() { return _right; }
        void set_right(std::shared_ptr<SPTNode> right) { _right = right; }

        std::shared_ptr<SPTNode> parent() { return _parent; }
        void set_parent(std::shared_ptr<SPTNode> parent) { _parent = parent; }

        Eigen::VectorXd max() { return _max; }
        void set_max(const Eigen::VectorXd& max) { _max = max; }

        Eigen::VectorXd min() { return _min; }
        void set_min(const Eigen::VectorXd& min) { _min = min; }

        int _depth;

    protected:
        std::shared_ptr<SPTNode> _parent, _left, _right;
        std::vector<Eigen::VectorXd> _points;
        Eigen::VectorXd _split_dir, _split_vector;
        Eigen::VectorXd _max, _min;
        double _split_median;
    };

    inline Eigen::MatrixXd sample_covariance(const std::vector<Eigen::VectorXd>& points)
    {
        assert(points.size());

        // Get the sample means
        Eigen::VectorXd means = Eigen::VectorXd::Zero(points[0].size());

        for (int i = 0; i < points.size(); i++) {
            // for (size_t j = 0; j < points.size(); j++) {
            //     means(i) += points[j][i];
            // }
            // means(i) /= double(points.size());
            means.array() += points[i].array();
        }

        means = means.array() / double(points.size());

        // Calculate the sample covariance matrix
        Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(points[0].size(), points[0].size());
        for (int i = 0; i < points.size(); i++) {
            cov = cov + points[i] * points[i].transpose();
            // for (int j = 0; j < means.size(); j++) {
            //     for (size_t k = 0; k < points.size(); k++) {
            //         cov(i, j) = (points[k][i] - means(i)) * (points[k][j] - means(j));
            //     }
            //     cov(i, j) /= double(points.size() - 1);
            // }
        }

        cov = (cov.array() - (double(points.size()) * means * means.transpose()).array()) / (double(points.size()) - 1.0);

        // Eigen::EigenSolver<Eigen::MatrixXd> solver(cov);
        // Eigen::MatrixXcd eig = solver.eigenvectors();
        // std::cout << eig.col(0)[0].real() << " " << eig.col(0)[1].real() << " -> " << std::sqrt((eig.col(0)[0].real() * eig.col(0)[0].real()) + (eig.col(0)[1].real() * eig.col(0)[1].real())) << std::endl;
        // std::cout << eig.col(1)[0].real() << " " << eig.col(1)[1].real() << " -> " << std::sqrt((eig.col(1)[0].real() * eig.col(1)[0].real()) + (eig.col(1)[1].real() * eig.col(1)[1].real())) << std::endl;
        // std::cout << std::endl;

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

        double median;
        // for (int i = 0; i < medians.size(); i++) {
        std::vector<double> p_i;
        for (size_t j = 0; j < size; j++)
            p_i.push_back(points[j]);

        std::sort(p_i.begin(), p_i.end());

        median = (size % 2) ? p_i[size / 2] : (p_i[size / 2 - 1] + p_i[size / 2]) / 2;
        // }

        return median;
    }

    inline std::shared_ptr<SPTNode> make_spt(const std::vector<Eigen::VectorXd>& points, int max_depth = 2, int depth = 0)
    {
        auto spt_node = std::make_shared<SPTNode>(points);
        spt_node->_depth = depth;

        if (max_depth == 0) {
            return spt_node;
        }

        // calculate split direction
        Eigen::VectorXd split_dir = get_split_dir(points, depth);

        // calculate median
        std::vector<double> transformed_points = transform_points(points, split_dir);
        double split_median = get_median(transformed_points);

        spt_node->set_split_dir(split_dir);
        spt_node->set_split_median(split_median);

        // get new points
        double min = std::numeric_limits<double>::max();
        int min_i = -1;
        std::vector<Eigen::VectorXd> left_points, right_points;

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

            if (transformed_points[i] <= split_median) {
                left_points.push_back(points[i]);
                // this is needed for the boundaries computation
                for (int j = 0; j < points[0].size(); j++) {
                    if (points[i](j) < min_left(j)) {
                        min_left(j) = points[i](j);
                    }
                    if (points[i](j) > max_left(j)) {
                        max_left(j) = points[i](j);
                    }
                }

                // double dist = std::abs(transformed_points[i] - split_median);
                // if (dist < min_left2) {
                //     min_left2 = dist;
                //     min_i_left = i;
                // }
            }
            else {
                right_points.push_back(points[i]);
                // this is needed for the boundaries computation
                for (int j = 0; j < points[0].size(); j++) {
                    if (points[i](j) < min_right(j)) {
                        min_right(j) = points[i](j);
                    }
                    if (points[i](j) > max_right(j)) {
                        max_right(j) = points[i](j);
                    }
                }

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

        // Make left node
        auto left_node = make_spt(left_points, max_depth - 1, depth + 1);
        left_node->set_parent(spt_node);
        spt_node->set_left(left_node);

        // left_node->set_max(max_left);
        // left_node->set_min(min_left);
        // left_node->set_split_vector(points[min_i_left]);

        // Make right node
        auto right_node = make_spt(right_points, max_depth - 1, depth + 1);
        right_node->set_parent(spt_node);
        spt_node->set_right(right_node);

        // right_node->set_max(max_right);
        // right_node->set_min(min_right);
        // right_node->set_split_vector(points[min_i_right]);

        return spt_node;
    }

    inline std::vector<std::shared_ptr<SPTNode>> get_leaves(const std::shared_ptr<SPTNode>& spt_node)
    {
        std::vector<std::shared_ptr<SPTNode>> leaves;

        std::stack<std::shared_ptr<SPTNode>> S;
        S.push(spt_node);

        while (!S.empty()) {
            auto n = S.top();
            S.pop();

            auto l = n->left();
            auto r = n->right();

            if (l->left() == nullptr || l->right() == nullptr) {
                // std::cout << "Adding left" << std::endl;
                leaves.push_back(l);
            }
            else
                S.push(l);

            if (r->left() == nullptr || r->right() == nullptr) {
                // std::cout << "Adding right" << std::endl;
                leaves.push_back(r);
            }
            else
                S.push(r);
        }

        return leaves;
    }
}

#endif
