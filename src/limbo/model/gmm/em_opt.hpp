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
#ifndef LIMBO_MODEL_GMM_EM_OPT_HPP
#define LIMBO_MODEL_GMM_EM_OPT_HPP

#include <limbo/model/gp/hp_opt.hpp>

namespace limbo {
    namespace defaults {
        struct opt_gmm_em {
            /// @ingroup opt_gmm_em_defaults

            // maximum iterations, -1 for convergence
            BO_PARAM(int, max_iters, -1);

            // convergence epsilon, <= 0 for ignoring
            BO_PARAM(double, epsilon, 1e-3);
        };
    } // namespace defaults

    namespace model {
        namespace gmm {
            ///@ingroup model_opt
            ///optimize a GMM using Expectation-Maximization
            template <typename Params>
            struct EMOpt : public limbo::model::gp::HPOpt<Params> {
            public:
                EMOpt() { this->_called = true; }
                template <typename GMM>
                void operator()(GMM& gmm)
                {
                    int K = gmm.K();
                    int N = gmm.data().rows();
                    int N_iters = Params::opt_gmm_em::max_iters();
                    if (N_iters < 0)
                        N_iters = std::numeric_limits<int>::max();

                    double prev_lik = -std::numeric_limits<double>::max();

                    for (int n = 0; n < N_iters; n++) {
                        Eigen::VectorXd theta = gmm.params();
                        // calculate weights
                        Eigen::MatrixXd w = Eigen::MatrixXd::Zero(N, K);
                        // for each sample
                        for (int i = 0; i < N; i++) {
                            // for each mixture
                            for (int k = 0; k < K; k++) {
                                double pi_k = theta(k);

                                Eigen::VectorXd x = gmm.data().row(i);
                                double w_ik = pi_k * gmm.models()[k].prob(x);

                                double w_all = w_ik;
                                for (int j = 0; j < K; j++) {
                                    if (j == k)
                                        continue;
                                    double pi_j = theta(j);
                                    w_all += pi_j * gmm.models()[j].prob(x);
                                }

                                w(i, k) = w_ik / w_all;
                            }
                        }

                        // update parameters
                        for (int k = 0; k < K; k++) {
                            double N_k = w.col(k).sum();
                            theta(k) = N_k / static_cast<double>(N);
                            gmm.weights()[k] = theta(k);

                            Eigen::VectorXd mu_k = Eigen::VectorXd::Zero(gmm.data().cols());
                            Eigen::MatrixXd S_k = Eigen::MatrixXd::Identity(gmm.data().cols(), gmm.data().cols());
                            for (int i = 0; i < N; i++) {
                                Eigen::VectorXd x = gmm.data().row(i);

                                mu_k.array() += w(i, k) * x.array();
                            }

                            mu_k.array() /= (N_k + 1e-50);

                            for (int i = 0; i < N; i++) {
                                Eigen::VectorXd x = gmm.data().row(i);

                                S_k += w(i, k) * (x - mu_k) * (x - mu_k).transpose();
                            }

                            S_k.array() /= (N_k + 1e-50);

                            // update theta vector
                            // theta.segment(K + k * n_params, n_params) = _to_params(mu_k, S_k);
                            gmm.models()[k].mu() = mu_k;
                            gmm.models()[k].sigma() = S_k;
                        }

                        // log likelihood for stopping criteria
                        double log_lik = 0.;
                        for (int i = 0; i < N; i++) {
                            Eigen::VectorXd x = gmm.data().row(i);

                            double prob = 0.;
                            for (int k = 0; k < K; k++) {
                                double pi_k = theta(k);

                                prob += pi_k * gmm.models()[k].prob(x);
                            }

                            log_lik += std::log(prob);
                        }

                        std::cout << "loglik: " << log_lik << std::endl;

                        if (std::abs(prev_lik - log_lik) < Params::opt_gmm_em::epsilon())
                            break;

                        prev_lik = log_lik;
                    }
                }
            };

        } // namespace gmm
    } // namespace model
} // namespace limbo

#endif