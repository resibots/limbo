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
#ifndef LIMBO_MODEL_RBF_HPP
#define LIMBO_MODEL_RBF_HPP

#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/opt/rprop.hpp>

#include <limbo/model/rbf/kmeans.hpp>

namespace limbo {
    namespace defaults {
        struct model_rbf {
            /// @ingroup model_defaults
            BO_PARAM(int, N, 50);
            /// @ingroup model_defaults
            BO_PARAM(int, batch, 25);
            // /// @ingroup model_defaults
            // BO_PARAM(bool, normalized, false);
            /// @ingroup model_defaults
            BO_PARAM(bool, optimize_centers, false); // if optimized is set to false, we use K-Means to identify the centers
        };
    } // namespace defaults

    namespace model {
        /// @ingroup model
        /// A Radial basis function (RBF) network; a modified version to output predictive variance
        /// It is parametrized by:
        /// - a kernel function
        /// - a mean function
        /// - an optimizer for training
        template <typename Params, typename KernelFunction = kernel::SquaredExpARD<Params>, typename MeanFunction = mean::Data<Params>, typename ParamsOptimizer = opt::Rprop<Params>>
        class RBFNet {
        public:
            /// Single-Output simple RBF
            /// https://en.wikipedia.org/wiki/Radial_basis_function_network
            struct RBF {
                RBF(int N, int dim) : _kernel(KernelFunction(dim)), _weights(Eigen::VectorXd::Constant(N, 1. / N))
                {
                    _centers.resize(N, Eigen::VectorXd::Zero(dim));
                }

                double compute(const Eigen::VectorXd& x) const
                {
                    // TO-DO: Optimize this
                    double val = 0.;
                    for (int i = 0; i < _weights.size(); i++) {
                        val += _weights(i) * _kernel(x, _centers[i]);
                    }

                    return val;
                }

                Eigen::VectorXd gradient(const Eigen::VectorXd& x) const
                {
                    Eigen::VectorXd g = Eigen::VectorXd::Zero(params_size());

                    for (int i = 0; i < _weights.size(); i++) {
                        g.head(_kernel.h_params_size()).array() += _weights(i) * _kernel.grad(x, _centers[i]).array();
                    }

                    for (int i = 0; i < _weights.size(); i++) {
                        g.tail(_weights.size())[i] = _kernel(x, _centers[i]);
                    }

                    return g;
                }

                int params_size() const
                {
                    int size = _kernel.h_params_size() + _weights.size();
                    // if (Params::model_rbf::optimize_centers())
                    //     size += _centers.size() * _weights.size();
                    return size;
                }

                Eigen::VectorXd params() const
                {
                    Eigen::VectorXd p(params_size());
                    p.head(_kernel.h_params_size()) = _kernel.h_params();
                    p.tail(_weights.size()) = _weights;

                    return p;
                }

                void set_params(const Eigen::VectorXd& p)
                {
                    _kernel.set_h_params(p.head(_kernel.h_params_size()));
                    _weights = p.tail(_weights.size());
                }

                KernelFunction _kernel;
                Eigen::VectorXd _weights;
                std::vector<Eigen::VectorXd> _centers;
            };

            /// useful because the model might be created before knowing anything about the process
            RBFNet() : _dim_in(-1), _dim_out(-1) {}

            /// useful because the model might be created before having samples
            RBFNet(int dim_in, int dim_out)
                : _dim_in(dim_in), _dim_out(dim_out), _mean_function(dim_out)
            {
                _mean_rbfs.resize(_dim_out, RBF(Params::model_rbf::N(), _dim_in));
                _sigma_rbfs.resize(_dim_out, RBF(Params::model_rbf::N(), _dim_in));
            }

            /// Compute the GP from samples and observations. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations, bool full_training = true)
            {
                assert(samples.size() != 0);
                assert(observations.size() != 0);
                assert(samples.size() == observations.size());

                if (_dim_in != samples[0].size()) {
                    _dim_in = samples[0].size();
                }

                if (_dim_out != observations[0].size()) {
                    _dim_out = observations[0].size();
                    _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low
                }

                _samples = samples;
                _centers.resize(Params::model_rbf::N(), Eigen::VectorXd::Zero(_dim_in));

                _observations.resize(observations.size(), _dim_out);
                for (int i = 0; i < _observations.rows(); ++i)
                    _observations.row(i) = observations[i];

                _mean_observation = _observations.colwise().mean();

                if (static_cast<int>(_mean_rbfs.size()) != _dim_out) {
                    // _mean_rbfs.resize(_dim_out, RBF(&_centers, Params::model_rbf::N(), _dim_in));
                    _mean_rbfs.resize(_dim_out, RBF(Params::model_rbf::N(), _dim_in));
                }

                if (static_cast<int>(_sigma_rbfs.size()) != _dim_out) {
                    // _sigma_rbfs.resize(_dim_out, RBF(&_centers, Params::model_rbf::N(), _dim_in));
                    _sigma_rbfs.resize(_dim_out, RBF(Params::model_rbf::N(), _dim_in));
                }

                this->_compute_obs_mean();

                if (!Params::model_rbf::optimize_centers())
                    this->_compute_kmeans_centroids();

                if (full_training)
                    this->_train();
            }

            std::tuple<Eigen::VectorXd, Eigen::VectorXd> query(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return std::make_tuple(_mean_function(v, *this), Eigen::VectorXd::Zero(_dim_out));
                // _kernel_function(v, v) + _kernel_function.noise());

                Eigen::VectorXd mu(_dim_out);
                Eigen::VectorXd log_sigma(_dim_out);

                for (size_t i = 0; i < _mean_rbfs.size(); i++) {
                    mu(i) = _mean_rbfs[i].compute(v);
                    log_sigma(i) = _sigma_rbfs[i].compute(v);
                }

                return {mu + _mean_function(v, *this), log_sigma};
            }

            /// return the centroids used
            const std::vector<Eigen::VectorXd>& centroids() const { return _centers; }

            /// return the number of dimensions of the input
            int dim_in() const
            {
                assert(_dim_in != -1); // need to compute first!
                return _dim_in;
            }

            /// return the number of dimensions of the output
            int dim_out() const
            {
                assert(_dim_out != -1); // need to compute first!
                return _dim_out;
            }

            // const KernelFunction& kernel_function() const { return _kernel_function; }
            // KernelFunction& kernel_function() { return _kernel_function; }

            const MeanFunction& mean_function() const { return _mean_function; }
            MeanFunction& mean_function() { return _mean_function; }

            /// return the maximum observation (only call this if the output of the GP is of dimension 1)
            Eigen::VectorXd max_observation() const
            {
                if (_observations.cols() > 1)
                    std::cout << "WARNING max_observation with multi dimensional "
                                 "observations doesn't make sense"
                              << std::endl;
                return tools::make_vector(_observations.maxCoeff());
            }

            /// return the mean observation (only call this if the output of the GP is of dimension 1)
            Eigen::VectorXd mean_observation() const
            {
                assert(_dim_out > 0);
                return _samples.size() > 0 ? _mean_observation
                                           : Eigen::VectorXd::Zero(_dim_out);
            }

            const Eigen::MatrixXd& mean_vector() const { return _mean_vector; }

            const Eigen::MatrixXd& obs_mean() const { return _obs_mean; }

            /// return the number of samples used to compute the GP
            int nb_samples() const { return _samples.size(); }

            /// return the list of samples
            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

            /// return the list of observations
            std::vector<Eigen::VectorXd> observations() const
            {
                std::vector<Eigen::VectorXd> observations;
                for (int i = 0; i < _observations.rows(); i++) {
                    observations.push_back(_observations.row(i));
                }

                return observations;
            }

            /// return the observations (in matrix form)
            /// (NxD), where N is the number of points and D is the dimension output
            const Eigen::MatrixXd& observations_matrix() const
            {
                return _observations;
            }

        protected:
            int _dim_in;
            int _dim_out;

            // KernelFunction _kernel_function;
            MeanFunction _mean_function;

            std::vector<Eigen::VectorXd> _samples;
            std::vector<Eigen::VectorXd> _centers;
            Eigen::MatrixXd _observations;
            Eigen::MatrixXd _mean_vector;
            Eigen::MatrixXd _obs_mean;
            Eigen::VectorXd _mean_observation;

            std::vector<RBF> _mean_rbfs, _sigma_rbfs;

            void _compute_obs_mean()
            {
                assert(!_samples.empty());
                _mean_vector.resize(_samples.size(), _dim_out);
                for (int i = 0; i < _mean_vector.rows(); i++) {
                    assert(_samples[i].cols() == 1);
                    assert(_samples[i].rows() != 0);
                    assert(_samples[i].rows() == _dim_in);
                    _mean_vector.row(i) = _mean_function(_samples[i], *this);
                }
                _obs_mean = _observations - _mean_vector;
            }

            void _compute_kmeans_centroids()
            {
                _centers = rbf::kmeans(_samples, Params::model_rbf::N(), _dim_in);

                // COPYING FOR THREAD-SAFETY
                for (size_t i = 0; i < _mean_rbfs.size(); i++) {
                    _mean_rbfs[i]._centers = _centers;
                    _sigma_rbfs[i]._centers = _centers;
                }
            }

            void _train()
            {
                int rbf_p_size = _mean_rbfs[0].params_size();
                int param_size = 2 * _mean_rbfs.size() * rbf_p_size;
                if (Params::model_rbf::optimize_centers())
                    param_size += _dim_in * Params::model_rbf::N();

                auto get_params = [&]() {
                    Eigen::VectorXd p(param_size);

                    for (size_t i = 0; i < _mean_rbfs.size(); i++) {
                        p.segment(i * rbf_p_size, rbf_p_size) = _mean_rbfs[i].params();
                        p.segment(_mean_rbfs.size() * rbf_p_size + i * rbf_p_size, rbf_p_size) = _sigma_rbfs[i].params();
                    }

                    if (Params::model_rbf::optimize_centers()) {
                        for (size_t i = 0; i < _centers.size(); i++) {
                            p.segment(2 * _mean_rbfs.size() * rbf_p_size + i * _dim_in, _dim_in) = _centers[i];
                        }
                    }

                    return p;
                };

                auto get_grads = [&](const RBFNet& net, const Eigen::VectorXd& x) {
                    Eigen::VectorXd p(param_size);

                    for (size_t i = 0; i < net._mean_rbfs.size(); i++) {
                        p.segment(i * rbf_p_size, rbf_p_size) = net._mean_rbfs[i].gradient(x);
                        p.segment(net._mean_rbfs.size() * rbf_p_size + i * rbf_p_size, rbf_p_size) = net._sigma_rbfs[i].gradient(x);
                    }

                    if (Params::model_rbf::optimize_centers()) {
                        // for (size_t i = 0; i < _centers.size(); i++) {
                        //     p.segment(2 * _mean_rbfs.size() * rbf_p_size + i * _dim_in, _dim_in) = _centers[i];
                        // }
                        // TO-DO!
                    }

                    return p;
                };

                auto set_params = [&](RBFNet& net, const Eigen::VectorXd& p) {
                    for (size_t i = 0; i < net._mean_rbfs.size(); i++) {
                        net._mean_rbfs[i].set_params(p.segment(i * rbf_p_size, rbf_p_size));
                        net._sigma_rbfs[i].set_params(p.segment(net._mean_rbfs.size() * rbf_p_size + i * rbf_p_size, rbf_p_size));
                    }

                    if (Params::model_rbf::optimize_centers()) {
                        for (size_t i = 0; i < net._centers.size(); i++) {
                            net._centers[i] = p.segment(2 * _mean_rbfs.size() * rbf_p_size + i * _dim_in, _dim_in);
                        }

                        // COPYING FOR THREAD-SAFETY
                        for (size_t i = 0; i < net._mean_rbfs.size(); i++) {
                            net._mean_rbfs[i]._centers = net._centers;
                            net._sigma_rbfs[i]._centers = net._centers;
                        }
                    }
                };

                Eigen::VectorXd init_params = get_params();

                auto loss_func = [&](const Eigen::VectorXd& params, bool compute_grad) {
                    RBFNet net = *this;

                    set_params(net, params);

                    Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());

                    double loss = 0.;
                    for (size_t i = 0; i < _samples.size(); i++) {
                        Eigen::VectorXd gradient = get_grads(net, _samples[i]);

                        // Eigen::VectorXd mu, log_sigma;
                        // std::tie(mu, log_sigma) = net.query(_samples[i]);
                        Eigen::VectorXd mu(_dim_out);
                        Eigen::VectorXd log_sigma(_dim_out);

                        for (size_t j = 0; j < net._mean_rbfs.size(); j++) {
                            mu(j) = net._mean_rbfs[j].compute(_samples[i]);
                            log_sigma(j) = net._sigma_rbfs[j].compute(_samples[i]);
                        }

                        Eigen::VectorXd sigma = log_sigma.array().exp();

                        Eigen::VectorXd diff = mu - _obs_mean.row(i);
                        Eigen::VectorXd diff_sq = diff.array().square();
                        Eigen::VectorXd inv_sigma = 1. / (sigma.array() + 1e-20);
                        double logdet_sigma = log_sigma.sum();

                        Eigen::MatrixXd inv_S = inv_sigma.asDiagonal();

                        loss += diff.transpose() * inv_S * diff + logdet_sigma;

                        // grad for mean and sigma
                        Eigen::VectorXd tmp_mean = 2. * diff.array() * inv_sigma.array();
                        Eigen::VectorXd tmp_sigma = -diff_sq.array() * inv_sigma.array();
                        for (size_t j = 0; j < net._mean_rbfs.size(); j++) {
                            // mean
                            // grad.segment(j * rbf_p_size, rbf_p_size).array() += (2. * diff.array() * inv_sigma.array()) * gradient.segment(j * rbf_p_size, rbf_p_size).array();
                            for (int k = 0; k < _dim_out; k++)
                                grad.segment(j * rbf_p_size, rbf_p_size).array() += tmp_mean[k] * gradient.segment(j * rbf_p_size, rbf_p_size).array();

                            // sigma
                            // grad.segment(net._mean_rbfs.size() * rbf_p_size + i * rbf_p_size, rbf_p_size).array() += (-diff_sq.array() * inv_sigma.array()) * gradient.segment(net._mean_rbfs.size() * rbf_p_size + i * rbf_p_size, rbf_p_size).array();
                            for (int k = 0; k < _dim_out; k++)
                                grad.segment(net._mean_rbfs.size() * rbf_p_size + j * rbf_p_size, rbf_p_size).array() += tmp_sigma[k] * gradient.segment(net._mean_rbfs.size() * rbf_p_size + j * rbf_p_size, rbf_p_size).array();

                            // grad for logdet-sigma
                            grad.segment(net._mean_rbfs.size() * rbf_p_size + j * rbf_p_size, rbf_p_size).array() += gradient.segment(net._mean_rbfs.size() * rbf_p_size + j * rbf_p_size, rbf_p_size).array();
                        }
                    }

                    grad.array() = grad.array() / double(_samples.size());

                    return opt::eval_t{-loss / double(_samples.size()), -grad};
                };

                ParamsOptimizer opt;

                Eigen::VectorXd best_params = opt(loss_func, init_params, false);

                set_params(*this, best_params);
            }
        };
    } // namespace model
} // namespace limbo

#endif
