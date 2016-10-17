//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
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
#ifndef LIMBO_MODEL_GP_HPP
#define LIMBO_MODEL_GP_HPP

#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>

#include <limbo/model/gp/no_lf_opt.hpp>
#include <limbo/tools.hpp>

namespace limbo {
    namespace model {
        /// @ingroup model
        /// A classic Gaussian process.
        /// It is parametrized by:
        /// - a mean function
        /// - [optionnal] an optimizer for the hyper-parameters
        template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = gp::NoLFOpt<Params>>
        class GP {
        public:
            /// useful because the model might be created before knowing anything about the process
            GP() : _dim_in(-1), _dim_out(-1) {}

            /// useful because the model might be created  before having samples
            GP(int dim_in, int dim_out)
                : _dim_in(dim_in), _dim_out(dim_out), _kernel_function(dim_in), _mean_function(dim_out) {}

            /// Compute the GP from samples, observation, noise. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations,
                const Eigen::VectorXd& noises, bool compute_kernel = true)
            {
                assert(samples.size() != 0);
                assert(observations.size() != 0);
                assert(samples.size() == observations.size());

                _dim_in = samples[0].size();
                _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low

                _dim_out = observations[0].size();
                _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low

                _samples = samples;

                _observations.resize(observations.size(), _dim_out);
                for (int i = 0; i < _observations.rows(); ++i)
                    _observations.row(i) = observations[i];

                _mean_observation = _observations.colwise().mean();

                _noises = noises;

                this->_compute_obs_mean();
                if (compute_kernel)
                    this->_compute_full_kernel();
            }

            /// Do not forget to call this if you use hyper-prameters optimization!!
            void optimize_hyperparams()
            {
                _hp_optimize(*this);
            }

            /// add sample and update the GP. This code uses an incremental implementation of the Cholesky
            /// decomposition. It is therefore much faster than a call to compute()
            void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation, double noise)
            {
                if (_samples.empty()) {
                    _dim_in = sample.size();
                    _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low

                    _dim_out = observation.size();
                    _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low
                }
                else {
                    assert(sample.size() == _dim_in);
                    assert(observation.size() == _dim_out);
                }

                _samples.push_back(sample);

                _observations.conservativeResize(_observations.rows() + 1, _dim_out);
                _observations.bottomRows<1>() = observation.transpose();

                _mean_observation = _observations.colwise().mean();

                _noises.conservativeResize(_noises.size() + 1);
                _noises[_noises.size() - 1] = noise;
                //_noise = noise;

                this->_compute_obs_mean();
                this->_compute_incremental_kernel();
            }

            /**
             \\rst
             return :math:`\mu`, :math:`\sigma^2` (unormalized). If there is no sample, return the value according to the mean function. Using this method instead of separate calls to mu() and sigma() is more efficient because some computations are shared between mu() and sigma().
             \\endrst
	  		*/
            std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return std::make_tuple(_mean_function(v, *this),
                        _kernel_function(v, v));

                Eigen::VectorXd k = _compute_k(v);
                return std::make_tuple(_mu(v, k), _sigma(v, k));
            }

            /**
             \\rst
             return :math:`\mu` (unormalized). If there is no sample, return the value according to the mean function.
             \\endrst
	  		*/
            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return _mean_function(v, *this);
                return _mu(v, _compute_k(v));
            }

            /**
             \\rst
             return :math:`\sigma^2` (unormalized). If there is no sample, return the max :math:`\sigma^2`.
             \\endrst
	  		*/
            double sigma(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return _kernel_function(v, v);
                return _sigma(v, _compute_k(v));
            }

            /// return the number of dimensions of the input
            int dim_in() const
            {
                assert(_dim_in != -1); // need to compute first !
                return _dim_in;
            }

            /// return the number of dimensions of the output
            int dim_out() const
            {
                assert(_dim_out != -1); // need to compute first !
                return _dim_out;
            }

            const KernelFunction& kernel_function() const { return _kernel_function; }

            KernelFunction& kernel_function() { return _kernel_function; }

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
                // TODO: Check if _dim_out is correct?!
                return _samples.size() > 0 ? _mean_observation
                                           : Eigen::VectorXd::Zero(_dim_out);
            }

            const Eigen::MatrixXd& mean_vector() const { return _mean_vector; }

            const Eigen::MatrixXd& obs_mean() const { return _obs_mean; }

            /// return the number of samples used to compute the GP
            int nb_samples() const { return _samples.size(); }

            ///  recomputes the GP
            void recompute(bool update_obs_mean = true)
            {
                assert(!_samples.empty());

                if (update_obs_mean)
                    this->_compute_obs_mean();

                this->_compute_full_kernel();
            }

            /// recomputes the internal variable related to the mean object without recomputing the kernel
            void recompute_mean_internal()
            {
                assert(!_samples.empty());
                this->_compute_obs_mean();
                this->_compute_alpha();
            }

            /// return the likelihood (do not compute it!)
            double get_lik() const { return _lik; }

            /// set the likelihood (you need to compute it from outside!)
            void set_lik(const double& lik) { _lik = lik; }

            /// LLT matrix (from Cholesky decomposition)
            //const Eigen::LLT<Eigen::MatrixXd>& llt() const { return _llt; }
            const Eigen::MatrixXd& matrixL() const { return _matrixL; }

            const Eigen::MatrixXd& alpha() const { return _alpha; }

            /// return the list of samples that have been tested so far
            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

        protected:
            int _dim_in;
            int _dim_out;

            KernelFunction _kernel_function;
            MeanFunction _mean_function;

            std::vector<Eigen::VectorXd> _samples;
            Eigen::MatrixXd _observations;
            Eigen::MatrixXd _mean_vector;
            Eigen::MatrixXd _obs_mean;

            Eigen::VectorXd _noises;
            Eigen::VectorXd _noises_bl;

            Eigen::MatrixXd _alpha;
            Eigen::VectorXd _mean_observation;

            Eigen::MatrixXd _kernel;

            Eigen::MatrixXd _matrixL;

            double _lik;

            HyperParamsOptimizer _hp_optimize;

            void _compute_obs_mean()
            {
                _mean_vector.resize(_samples.size(), _dim_out);
                for (int i = 0; i < _mean_vector.rows(); i++)
                    _mean_vector.row(i) = _mean_function(_samples[i], *this);
                _obs_mean = _observations - _mean_vector;
            }

            void _compute_full_kernel()
            {
                size_t n = _samples.size();
                _kernel.resize(n, n);

                // O(n^2) [should be negligible]
                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j <= i; ++j)
                        _kernel(i, j) = _kernel_function(_samples[i], _samples[j]) + ((i == j) ? _noises[i] : 0); // noise only on the diagonal

                for (size_t i = 0; i < n; i++)
                    for (size_t j = 0; j < i; ++j)
                        _kernel(j, i) = _kernel(i, j);

                // O(n^3)
                _matrixL = Eigen::LLT<Eigen::MatrixXd>(_kernel).matrixL();

                this->_compute_alpha();
            }

            void _compute_incremental_kernel()
            {
                // Incremental LLT
                // This part of the code is inpired from the Bayesopt Library (cholesky_add_row function).
                // However, the mathematical fundations can be easily retrieved by detailling the equations of the
                // extended L matrix that produces the desired kernel.

                size_t n = _samples.size();
                _kernel.conservativeResize(n, n);

                for (size_t i = 0; i < n; ++i) {
                    _kernel(i, n - 1) = _kernel_function(_samples[i], _samples[n - 1]) + ((i == n - 1) ? _noises[i] : 0); // noise only on the diagonal
                    _kernel(n - 1, i) = _kernel(i, n - 1);
                }

                _matrixL.conservativeResizeLike(Eigen::MatrixXd::Zero(n, n));

                double L_j;
                for (size_t j = 0; j < n - 1; ++j) {
                    L_j = _kernel(n - 1, j) - (_matrixL.block(j, 0, 1, j) * _matrixL.block(n - 1, 0, 1, j).transpose())(0, 0);
                    _matrixL(n - 1, j) = (L_j) / _matrixL(j, j);
                }

                L_j = _kernel(n - 1, n - 1) - (_matrixL.block(n - 1, 0, 1, n - 1) * _matrixL.block(n - 1, 0, 1, n - 1).transpose())(0, 0);
                _matrixL(n - 1, n - 1) = sqrt(L_j);

                this->_compute_alpha();
            }

            void _compute_alpha()
            {
                // alpha = K^{-1} * this->_obs_mean;
                Eigen::TriangularView<Eigen::MatrixXd, Eigen::Lower> triang = _matrixL.template triangularView<Eigen::Lower>();
                _alpha = triang.solve(_obs_mean);
                triang.adjoint().solveInPlace(_alpha);
            }

            Eigen::VectorXd _mu(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const
            {
                return (k.transpose() * _alpha) + _mean_function(v, *this).transpose();
            }

            double _sigma(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const
            {
                Eigen::VectorXd z = _matrixL.triangularView<Eigen::Lower>().solve(k);
                double res = _kernel_function(v, v) - z.dot(z);

                return (res <= std::numeric_limits<double>::epsilon()) ? 0 : res;
            }

            Eigen::VectorXd _compute_k(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd k(_samples.size());
                for (int i = 0; i < k.size(); i++)
                    k[i] = _kernel_function(_samples[i], v);
                return k;
            }
        };
    }
}

#endif
