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
#ifndef LIMBO_MODEL_GMM_KMEANS_HPP
#define LIMBO_MODEL_GMM_KMEANS_HPP

#include <map>
#include <vector>

#include <Eigen/Core>

// Quick hack for definition of 'I' in <complex.h>
#undef I

#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace defaults {
        struct KMeansPlusPlus {
            /// Initialize the centroids with the KMeans++ methodology
            Eigen::MatrixXd operator()(const Eigen::MatrixXd& data, int K) const
            {
                static thread_local tools::rgen_int_t rgeni(0, 1);
                static thread_local tools::rgen_double_t rgend(0, 1);

                // first centroid is a random point
                Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(1, data.cols());
                centroids.row(0) = data.row(rgeni.rand());

                // repeat until desired K is reached
                while (centroids.rows() < K) {
                    // compute the distances of every point to every centroid
                    // dist = [ ||data - centroid_1 ||^2, ..., ||data - centroid_n ||^2  ] -> NxK
                    Eigen::MatrixXd dist(data.rows(), centroids.rows());
                    for (int i = 0; i < centroids.rows(); i++) {
                        for (int j = 0; j < data.rows(); j++)
                            dist(j, i) = (data.row(j) - centroids.row(i)).squaredNorm();
                    }

                    // keep the minimum distances
                    // D = min(dist) -> Nx1
                    Eigen::MatrixXd D = dist.rowwise().minCoeff();

                    // compute the distribution D^2 / sum(D^2)
                    Eigen::MatrixXd probs = D / D.sum();
                    Eigen::MatrixXd cumprobs(probs.rows(), probs.cols());
                    cumprobs(0) = probs(0);
                    for (int i = 1; i < cumprobs.rows(); i++) {
                        cumprobs(i) = probs(i) + cumprobs(i - 1);
                    }

                    // select a new centroid according to the distribution computed
                    // in the previous step
                    int idx = -1;
                    double prob = rgend.rand();
                    for (int i = 1; i < cumprobs.rows(); i++) {
                        if (cumprobs(i) > prob) {
                            idx = i;
                            break;
                        }
                    }
                    assert(idx > 0);

                    // update centroids
                    centroids.conservativeResize(centroids.rows() + 1, centroids.cols());
                    centroids.row(centroids.rows() - 1) = data.row(idx);
                }
                return centroids;
            }
        };

        struct RandomInit {
            /// Initialize the centroids randomly
            Eigen::MatrixXd operator()(const Eigen::MatrixXd& data, int K) const
            {
                static thread_local tools::rgen_int_t rgen(0, 1);
                Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(K, data.cols());
                for (int i = 0; i < K; i++) {
                    centroids.row(i) = data.row(rgen.rand());
                }
                return centroids;
            }
        };
    } // namespace defaults

    namespace model {
        namespace clustering {

            template <typename InitFunc = defaults::KMeansPlusPlus>
            class KMeans {
            public:
                std::vector<Eigen::MatrixXd> fit(const Eigen::MatrixXd& data, const int K,
                    const int num_init = 5, const int max_iter = 100)
                {

                    _centroids = InitFunc()(data, K);
                    Eigen::MatrixXd prev_centroids;
                    _labels = Eigen::VectorXi::Ones(data.rows()) * -1;
                    for (int n = 0; n < max_iter; ++n) {
                        // assign points to cluster
                        _clusters.clear();
                        _clusters.resize(K);
                        _inertia = 0;

                        for (int i = 0; i < data.rows(); ++i) {
                            double min = std::numeric_limits<double>::max();
                            int min_k = -1;
                            for (int k = 0; k < K; k++) {
                                double dist = (_centroids.row(k) - data.row(i)).squaredNorm();
                                if (dist < min) {
                                    min = dist;
                                    min_k = k;
                                    _labels(i) = min_k;
                                    _inertia += dist;
                                }
                            }

                            _clusters[min_k].conservativeResize(_clusters[min_k].rows() + 1, data.cols());
                            _clusters[min_k].row(_clusters[min_k].rows() - 1) = data.row(i);
                        }
                        _inertia /= data.rows();

                        if (prev_centroids.size() && (prev_centroids == _centroids))
                            break; // algorithm has converged
                        else
                            prev_centroids = _centroids;

                        // update centroids
                        for (int k = 0; k < K; ++k) {
                            if (_clusters[k].size() == 0)
                                _centroids.row(k) = Eigen::VectorXd::Zero(data.cols());
                            else
                                _centroids.row(k) = _clusters[k].colwise().mean();
                        }
                    }

                    return _clusters;
                }

                Eigen::MatrixXi predict(const Eigen::MatrixXd& data) const
                {
                    assert(data.cols() == _centroids.cols());
                    Eigen::MatrixXi predictions(data.rows(), 1);
                    for (int i = 0; i < data.rows(); ++i) {
                        double min_d = std::numeric_limits<double>::max();
                        int min_k = -1;
                        for (int k = 0; k < _centroids.rows(); ++k) {
                            double dist = (_centroids.row(k) - data.row(i)).squaredNorm();
                            if (dist < min_d) {
                                min_d = dist;
                                min_k = k;
                            }
                        }
                        predictions(i) = min_k;
                    }
                    return predictions;
                }

                const Eigen::MatrixXd& centroids() const { return _centroids; }
                double inertia() const { return _inertia; }
                const Eigen::VectorXi& labels() const { return _labels; }

            protected:
                Eigen::MatrixXd _centroids;
                std::vector<Eigen::MatrixXd> _clusters;
                Eigen::VectorXi _labels;
                double _inertia;
            };

        } // namespace clustering
    } // namespace model
} // namespace limbo

#endif