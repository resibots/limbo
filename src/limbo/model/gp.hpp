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

            /// Compute the GP from samples, observation, noise. [optionnal: blacklisted samples]. This call needs to be explicit!
            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations, double noise,
                const std::vector<Eigen::VectorXd>& bl_samples = std::vector<Eigen::VectorXd>())
            {
                //should be checked each time! not only the first time
                assert(samples.size() != 0);
                assert(observations.size() != 0);
                assert(samples.size() == observations.size());

                if (_dim_in != samples[0].size()) {
                    _dim_in = samples[0].size();
                    _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low
                }

                if (_dim_out != observations[0].size()) {
                    _dim_out = observations[0].size();
                    _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low
                }

                _samples = samples;

                _observations.resize(observations.size(), _dim_out);
                for (int i = 0; i < _observations.rows(); ++i)
                    _observations.row(i) = observations[i];

                _mean_observation = _observations.colwise().mean();
                //_mean_observation.resize(_dim_out)
                //for (int i = 0; i < _dim_out; i++)
                //    _mean_observation(i) = _observations.col(i).mean();

                _noise = noise;

                _bl_samples = bl_samples;

                this->_compute_obs_mean();
                this->_compute_kernel(false);

                if (!_bl_samples.empty())
                    this->_compute_bl_kernel();

                HyperParamsOptimizer()(*this);
            }

            void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation, double noise)
            {
                if (_samples.empty()) {
                    if (_bl_samples.empty()) {
                        _dim_in = sample.size();
                        _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low
                    } else {
                        assert(sample.size() == _dim_in);
                    }

                    _dim_out = observation.size();
                    _mean_function = MeanFunction(_dim_out); // the cost of building a functor should be relatively low
                } else {
                    assert(sample.size() == _dim_in);
                    assert(observation.size() == _dim_out);
                }

                _samples.push_back(sample);

                _observations.conservativeResize(_observations.rows() + 1, _dim_out);
                _observations.bottomRows<1>() = observation.transpose();

                _mean_observation = _observations.colwise().mean();
                //_mean_observation.resize(_dim_out)
                //for (int i = 0; i < _dim_out; i++)
                //    _mean_observation(i) = _observations.col(i).mean();

                _noise = noise;

                this->_compute_obs_mean();
                this->_compute_kernel(true);

                if (!_bl_samples.empty())
                    this->_compute_bl_kernel();

                HyperParamsOptimizer()(*this);
            }

            void add_bl_sample(const Eigen::VectorXd& bl_sample, double noise)
            {
                if (_samples.empty() && _bl_samples.empty()) {
                    _dim_in = bl_sample.size();
                    _kernel_function = KernelFunction(_dim_in); // the cost of building a functor should be relatively low
                } else {
                    assert(bl_sample.size() == _dim_in);
                }

                _bl_samples.push_back(bl_sample);
                _noise = noise;

                if (!_samples.empty()) {
                    this->_compute_bl_kernel();
                    HyperParamsOptimizer()(*this);
                }
            }

            /**
             \\rst
             return :math:`\\mu`, :math:`\\sigma^2` (unormalized). If there is no sample, return the value according to the mean function. Using this method instead of separate calls to mu() and sigma() is more efficient because some computations are shared between mu() and sigma().
             \\endrst
	  		*/
            std::tuple<Eigen::VectorXd, double> query(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0 && _bl_samples.size() == 0)
                    return std::make_tuple(_mean_function(v, *this),
                        sqrt(_kernel_function(v, v)));

                if (_samples.size() == 0)
                    return std::make_tuple(_mean_function(v, *this),
                        _sigma(v, _compute_k_bl(v, _compute_k(v))));

                Eigen::VectorXd k = _compute_k(v);
                return std::make_tuple(_mu(v, k), _sigma(v, _compute_k_bl(v, k)));
            }

            /**
             \\rst
             return :math:`\\mu` (unormalized). If there is no sample, return the value according to the mean function.
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
             return :math:`\\sigma^2` (unormalized). If there is no sample, return the value according to the mean function.
             \\endrst
	  		*/
            double sigma(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0 && _bl_samples.size() == 0)
                    return sqrt(_kernel_function(v, v));
                return _sigma(v, _compute_k_bl(v, _compute_k(v)));
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
                return _observations.maxCoeff();
            }

            /// return the mean observation (only call this if the output of the GP is of dimension 1)
            Eigen::VectorXd mean_observation() const
            {
                // TO-DO: Check if _dim_out is correct?!
                return _samples.size() > 0 ? _mean_observation
                                           : Eigen::VectorXd::Zero(_dim_out);
            }

            const Eigen::MatrixXd& mean_vector() const { return _mean_vector; }

            const Eigen::MatrixXd& obs_mean() const { return _obs_mean; }

            /// return the number of samples used to compute the GP
            int nb_samples() const { return _samples.size(); }

            /** return the number of blacklisted samples used to compute the GP
	     \\rst
	     For the blacklist concept, see the Limbo-specific concept guide.
	     \\endrst
	     */
            int nb_bl_samples() const { return _bl_samples.size(); }

            ///  recomputes the GP
            void recompute(bool update_obs_mean)
            {
                assert(!_samples.empty());

                if (update_obs_mean)
                    this->_compute_obs_mean(); // ORDER MATTERS

                this->_compute_kernel(false);

                if (!_bl_samples.empty())
                    this->_compute_bl_kernel();
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
            std::vector<Eigen::VectorXd> _bl_samples; // black listed samples
            Eigen::MatrixXd _mean_vector;
            Eigen::MatrixXd _obs_mean;

            double _noise;
            Eigen::MatrixXd _alpha;
            Eigen::VectorXd _mean_observation;

            Eigen::MatrixXd _kernel;

            // Eigen::MatrixXd _inverted_kernel;

            Eigen::MatrixXd _matrixL;
            Eigen::MatrixXd _inv_bl_kernel;

            double _lik;

            void _compute_obs_mean()
            {
                _mean_vector.resize(_samples.size(), _dim_out);
                for (int i = 0; i < _mean_vector.rows(); i++)
                    _mean_vector.row(i) = _mean_function(_samples[i], *this);
                _obs_mean = _observations - _mean_vector;
            }

            void _compute_kernel(bool incremental = false)
            {
	      
                // Check if new kernel is the same as before
                if (!incremental) //_kernel.cols()==0 || !kernel.block(0,0,_kernel.rows(),_kernel.cols()).isApprox(_kernel, 1e-10)) // incremental LLT is impossible
                {
		  
		          _kernel.resize(_samples.size(), _samples.size());
		  
                    // O(n^2) [should be negligible]
                    for (size_t i = 0; i < _samples.size(); i++)
                        for (size_t j = 0; j < _samples.size(); ++j)
                            _kernel(i, j) = _kernel_function(_samples[i], _samples[j]) + ((i == j) ? _noise : 0); // noise only on the diagonal
		  
		  
                    // O(n^3)
                    //  _inverted_kernel = _kernel.inverse();
                    _matrixL = Eigen::LLT<Eigen::MatrixXd>(_kernel).matrixL();
		  
		    
                }
                else //incremental LLT
                //This part of the code is inpired from the Bayesopt Library (cholesky_add_row function). However, the mathematical fundations can be easily retrieved by detailling the equations of the extended L matrix that produces the desired kernel.
                {
		  
		  size_t n = _samples.size();
		  _kernel.conservativeResize(n, n);
		  

		  
                    for (size_t i = 0; i < n; ++i) {
                        _kernel(i, n - 1) = _kernel_function(_samples[i], _samples[n - 1]) + ((i == n - 1) ? _noise : 0); // noise only on the diagonal
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
		  
                }


 
                // alpha = K^{-1} * this->_obs_mean;
                _alpha = _matrixL.template triangularView<Eigen::Lower>().solve(_obs_mean);
                _matrixL.template triangularView<Eigen::Lower>().adjoint().solveInPlace(_alpha); //can probably be improved by avoiding to generate the view twice
						  
            }

            void _compute_bl_kernel()
            {
                Eigen::MatrixXd A1 = Eigen::MatrixXd::Identity(this->_samples.size(), this->_samples.size());
                _matrixL.template triangularView<Eigen::Lower>().solveInPlace(A1);
                _matrixL.template triangularView<Eigen::Lower>().transpose().solveInPlace(A1);

                _inv_bl_kernel.resize(_samples.size() + _bl_samples.size(),
                    _samples.size() + _bl_samples.size());

                Eigen::MatrixXd B(_samples.size(), _bl_samples.size());
                for (size_t i = 0; i < _samples.size(); i++)
                    for (size_t j = 0; j < _bl_samples.size(); ++j)
                        B(i, j) = _kernel_function(_samples[i], _bl_samples[j]);

                Eigen::MatrixXd D(_bl_samples.size(), _bl_samples.size());
                for (size_t i = 0; i < _bl_samples.size(); i++)
                    for (size_t j = 0; j < _bl_samples.size(); ++j)
                        D(i, j) = _kernel_function(_bl_samples[i], _bl_samples[j]) + ((i == j) ? _noise : 0);

                Eigen::MatrixXd comA = (D - B.transpose() * A1 * B);
                Eigen::LLT<Eigen::MatrixXd> llt_bl(comA);
                Eigen::MatrixXd comA1 = Eigen::MatrixXd::Identity(_bl_samples.size(), _bl_samples.size());
                llt_bl.matrixL().solveInPlace(comA1);
                llt_bl.matrixL().transpose().solveInPlace(comA1);

                // fill the matrix block wise
                _inv_bl_kernel.block(0, 0, _samples.size(), _samples.size()) = A1 + A1 * B * comA1 * B.transpose() * A1;
                _inv_bl_kernel.block(0, _samples.size(), _samples.size(),
                    _bl_samples.size())
                    = -A1 * B * comA1;
                _inv_bl_kernel.block(_samples.size(), 0, _bl_samples.size(),
                    _samples.size())
                    = _inv_bl_kernel.block(0, _samples.size(), _samples.size(),
                                        _bl_samples.size())
                          .transpose();
                _inv_bl_kernel.block(_samples.size(), _samples.size(), _bl_samples.size(),
                    _bl_samples.size())
                    = comA1;
            }

            Eigen::VectorXd _mu(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const
            {
                return (k.transpose() * _alpha) + _mean_function(v, *this).transpose();
            }

            double _sigma(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const
            {
                double res;
                if (_bl_samples.size() == 0) {
                    Eigen::VectorXd z = _matrixL.triangularView<Eigen::Lower>().solve(k);
                    res = _kernel_function(v, v) - z.dot(z);
                }
                else {
                    res = _kernel_function(v, v) - k.transpose() * _inv_bl_kernel * k;
                }

                return (res <= std::numeric_limits<double>::epsilon()) ? 0 : res;
            }

            Eigen::VectorXd _compute_k(const Eigen::VectorXd& v) const
            {
                Eigen::VectorXd k(_samples.size());
                for (int i = 0; i < k.size(); i++)
                    k[i] = _kernel_function(_samples[i], v);
                return k;
            }

            Eigen::VectorXd _compute_k_bl(const Eigen::VectorXd& v,
                const Eigen::VectorXd& k) const
            {
                if (_bl_samples.size() == 0) {
                    return k;
                }

                Eigen::VectorXd k_bl(_samples.size() + _bl_samples.size());

                k_bl.head(_samples.size()) = k;
                for (size_t i = 0; i < _bl_samples.size(); i++)
                    k_bl[i + this->_samples.size()] = this->_kernel_function(_bl_samples[i], v);
                return k_bl;
            }
        };
    }
}

#endif
