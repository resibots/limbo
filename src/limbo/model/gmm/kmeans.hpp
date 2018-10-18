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
    namespace model {
        namespace gmm {
            /// Cluster the data (NxD) in K clusters
            /// returns a vector with the clusters (NxD)
            std::vector<Eigen::MatrixXd> kmeans(const Eigen::MatrixXd& data, int K, int max_iter = 100)
            {
                static thread_local tools::rgen_int_t rgen(0, data.rows() - 1);

                Eigen::MatrixXd centroids = Eigen::MatrixXd::Zero(K, data.cols());
                // random points as centroids in the beginning
                for (int i = 0; i < K; i++) {
                    centroids.row(i) = data.row(rgen.rand());
                }

                std::vector<Eigen::MatrixXd> clusters;

                for (int n = 0; n < max_iter; n++) {
                    // assign points to cluster
                    clusters.clear();
                    clusters.resize(K);

                    for (int i = 0; i < data.rows(); i++) {
                        double min = std::numeric_limits<double>::max();
                        int min_k = -1;
                        for (int k = 0; k < K; k++) {
                            double dist = (centroids.row(k) - data.row(i)).squaredNorm();
                            if (dist < min) {
                                min = dist;
                                min_k = k;
                            }
                        }

                        clusters[min_k].conservativeResize(clusters[min_k].rows() + 1, data.cols());
                        clusters[min_k].row(clusters[min_k].rows() - 1) = data.row(i);
                    }

                    // update centroids
                    for (int k = 0; k < K; k++) {
                        if (clusters[k].size() == 0)
                            centroids.row(k) = Eigen::VectorXd::Zero(data.cols());
                        else
                            centroids.row(k) = clusters[k].colwise().mean();
                    }
                }

                return clusters;
            }
        } // namespace gmm
    } // namespace model
} // namespace limbo

#endif