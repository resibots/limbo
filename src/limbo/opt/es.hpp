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
#ifndef LIMBO_OPT_ES_HPP
#define LIMBO_OPT_ES_HPP

#include <algorithm>

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace defaults {
        struct opt_es {
            /// @ingroup opt_defaults
            /// size of population
            BO_PARAM(int, population, 50);

            /// @ingroup opt_defaults
            /// sigma_sq - exploration parameter
            BO_PARAM(double, sigma_sq, 0.1 * 0.1);

            /// @ingroup opt_defaults
            /// antithetic - turn on/off antithetic sampling
            BO_PARAM(bool, antithetic, true);

            /// @ingroup opt_defaults
            /// rank_fitness - use ranking as fitness instead of true fitness
            BO_PARAM(bool, rank_fitness, false);

            /// @ingroup opt_defaults
            /// normalize_fitness - normalize fitness (i.e., zero-mean, unit-variance)
            BO_PARAM(bool, normalize_fitness, false);

            /// @ingroup opt_defaults
            /// beta - gradient estimate multiplier
            BO_PARAM(double, beta, 2.);

            /// @ingroup opt_defaults
            /// alpha - approximate gradient information, [0,1]
            /// if set to 1: only ES
            /// if set to 0: only gradient
            BO_PARAM(double, alpha, 1.);

            /// @ingroup opt_defaults
            /// k - number of previous approx. gradients
            /// for orthonomal basis
            BO_PARAM(int, k, 2);
        };
    } // namespace defaults
    namespace opt {
        /// @ingroup opt
        /// Simple Evolutionary Strategies
        /// very close to finite differences gradient approximation
        /// It also implements "Guided evolutionary strategies" by Niru Maheswaranathan, Luke Metz, George Tucker, Jascha Sohl-Dickstein
        /// https://arxiv.org/abs/1806.10230
        ///
        /// Parameters:
        /// - int population
        /// - double sigma_sq
        /// - bool antithetic
        /// - bool rank_fitness
        /// - bool normalize_fitness
        /// - double beta
        /// - double alpha
        /// - int k
        /// - template parameter: Gradient-based optimizer
        template <typename Params, typename Optimizer>
        struct ES {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                assert(Params::opt_es::sigma_sq() > 0.);
                assert(Params::opt_es::alpha() >= 0. && Params::opt_es::alpha() <= 1.);
                assert(Params::opt_es::beta() > 0.);
                assert(Params::opt_es::k() > 0);
                assert(Params::opt_es::population() > 0 && (!Params::opt_es::antithetic() || Params::opt_es::population() % 2 == 0));

                size_t param_dim = init.size();
                int population = Params::opt_es::population();
                double sigma = std::sqrt(Params::opt_es::sigma_sq());
                bool antithetic = Params::opt_es::antithetic();
                bool rank_fitness = Params::opt_es::rank_fitness();
                bool normalize_fitness = Params::opt_es::normalize_fitness();
                double beta = Params::opt_es::beta();
                double alpha = Params::opt_es::alpha();
                int K = Params::opt_es::k();

                Eigen::MatrixXd approx_gradient = Eigen::MatrixXd::Zero(param_dim, K);
                Eigen::MatrixXd previous_grads = Eigen::MatrixXd::Zero(param_dim, 0);
                double es_term = std::sqrt(alpha / static_cast<double>(param_dim));
                double grad_term = std::sqrt((1. - alpha) / static_cast<double>(K));

                auto func = [&](const Eigen::VectorXd& params, bool eval_grad = false) {
                    Eigen::VectorXd vals(population);
                    Eigen::MatrixXd epsilons(population, param_dim);

                    // if we have approximate gradient
                    if (alpha < 1.) {
                        // get approximate gradient
                        auto perf = opt::eval_grad(f, params);
                        Eigen::VectorXd grad = opt::grad(perf);

                        previous_grads.conservativeResize(param_dim, previous_grads.cols() + 1);
                        previous_grads.col(previous_grads.cols() - 1) = grad;
                        if (previous_grads.cols() > K) {
                            _remove_column(previous_grads, 0);
                        }

                        if (previous_grads.cols() == K) {
                            approx_gradient = _gram_schmidt(previous_grads);
                        }
                    }

                    // generate random epsilons
                    for (int p = 0; p < population; p++) {
                        Eigen::VectorXd e = _gaussian_rand(Eigen::VectorXd::Zero(param_dim));
                        // if we have approximate gradient
                        if (alpha < 1. && previous_grads.cols() == K) {
                            e.array() *= es_term;
                            e.array() += grad_term * (approx_gradient * _gaussian_rand(Eigen::VectorXd::Zero(K))).array();
                        }
                        e *= sigma;
                        epsilons.row(p) = e;
                        if (antithetic) {
                            epsilons.row(p + 1) = -e;
                            p++;
                        }
                    }

                    // evaluate the population
                    tools::par::loop(0u, population, [&](size_t p) {
                        vals(p) = opt::eval(f, params.array() + epsilons.row(p).transpose().array());
                    });

                    // if we wish to use the rank as the fitness
                    if (rank_fitness) {
                        auto idx = _sort_indexes(vals);

                        for (int p = 0; p < vals.size(); p++)
                            vals[idx[p]] = p;
                        vals.array() /= static_cast<double>(vals.size() - 1.);
                        vals.array() -= 0.5;
                    }

                    // make the fitnesses Gaussian distributed
                    if (normalize_fitness)
                        vals = _normalize(vals);

                    // update the parameters
                    Eigen::VectorXd grad = beta * (epsilons.transpose() * vals).array() / (population * Params::opt_es::sigma_sq());
                    return std::make_pair(vals.mean(), grad);
                };

                Optimizer optimizer;

                return optimizer(func, init, bounded);
            }

        protected:
            std::vector<size_t> _sort_indexes(const Eigen::VectorXd& v) const
            {
                // initialize original index locations
                std::vector<size_t> idx(v.size());
                std::iota(idx.begin(), idx.end(), 0);

                // sort indexes based on comparing values in v
                std::sort(idx.begin(), idx.end(),
                    [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

                return idx;
            }

            Eigen::VectorXd _normalize(const Eigen::VectorXd& v) const
            {
                double mean = v.mean();
                double std_dev = std::sqrt((v.array() - mean).array().square().sum() / static_cast<double>(v.size() - 1));

                return (v.array() - mean) / std_dev;
            }

            Eigen::VectorXd _gaussian_rand(const Eigen::VectorXd& mean) const
            {
                static thread_local std::mt19937 gen(randutils::auto_seed_128{}.base());
                static thread_local std::normal_distribution<double> gaussian(0., 1.);

                Eigen::VectorXd result(mean.size());
                for (int i = 0; i < mean.size(); i++) {
                    result(i) = mean(i) + gaussian(gen);
                }

                return result;
            }

            Eigen::MatrixXd _gram_schmidt(const Eigen::MatrixXd& vectors) const
            {
                Eigen::MatrixXd v = Eigen::MatrixXd::Zero(vectors.rows(), vectors.cols());

                v.col(0) = vectors.col(0);

                for (int i = 1; i < v.cols(); i++) {
                    v.col(i) = vectors.col(i);
                    for (int j = 0; j < i; j++) {
                        v.col(i) -= _proj(vectors.col(i), v.col(j));
                    }
                }

                for (int i = 0; i < v.cols(); i++) {
                    v.col(i) = v.col(i).normalized();
                }

                return v;
            }

            Eigen::VectorXd _proj(const Eigen::VectorXd& v, const Eigen::VectorXd& u) const
            {
                return u.dot(v) / u.squaredNorm() * u;
            }

            void _remove_column(Eigen::MatrixXd& matrix, unsigned int colToRemove) const
            {
                unsigned int numRows = matrix.rows();
                unsigned int numCols = matrix.cols() - 1;

                if (colToRemove < numCols)
                    matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.block(0, colToRemove + 1, numRows, numCols - colToRemove);

                matrix.conservativeResize(numRows, numCols);
            }
        };
    } // namespace opt
} // namespace limbo

#endif
