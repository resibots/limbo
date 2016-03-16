#ifndef LIMBO_MODEL_GP_HPP
#define LIMBO_MODEL_GP_HPP

#include <iostream>
#include <cassert>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include <limbo/model/gp/no_lf_opt.hpp>

namespace limbo {
    namespace model {

        template <typename Params, typename KernelFunction, typename MeanFunction, class HyperParamsOptimizer = gp::NoLFOpt<Params>>
        class GP {
        public:
            GP() : _dim_in(-1), _dim_out(-1) {}

            // useful because the model might be created  before having samples
            GP(int dim_in, int dim_out)
                : _dim_in(dim_in), _dim_out(dim_out), _kernel_function(dim_in), _mean_function(dim_out) {}

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

                _mean_observation.resize(_dim_out);
                for (int i = 0; i < _observations.cols(); i++)
                    _mean_observation(i) = _observations.col(i).sum() / _observations.rows();

                _noise = noise;

                _bl_samples = bl_samples;

                _compute_obs_mean();
                _compute_kernel();

                HyperParamsOptimizer()(*this);
            }

            // return mu, sigma (unormaliz)
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

            Eigen::VectorXd mu(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0)
                    return _mean_function(v, *this);
                return _mu(v, _compute_k(v));
            }

            double sigma(const Eigen::VectorXd& v) const
            {
                if (_samples.size() == 0 && _bl_samples.size() == 0)
                    return sqrt(_kernel_function(v, v));
                return _sigma(v, _compute_k_bl(v, _compute_k(v)));
            }

            int dim_in() const
            {
                assert(_dim_in != -1); // need to compute first !
                return _dim_in;
            }

            int dim_out() const
            {
                assert(_dim_out != -1); // need to compute first !
                return _dim_out;
            }

            const KernelFunction& kernel_function() const { return _kernel_function; }

            KernelFunction& kernel_function() { return _kernel_function; }

            const MeanFunction& mean_function() const { return _mean_function; }

            MeanFunction& mean_function() { return _mean_function; }

            Eigen::VectorXd max_observation() const
            {
                if (_observations.cols() > 1)
                    std::cout << "WARNING max_observation with multi dimensional "
                                 "observations doesn't make sense" << std::endl;
                return _observations.maxCoeff();
            }

            Eigen::VectorXd mean_observation() const
            {
                // TO-DO: Check if _dim_out is correct?!
                return _samples.size() > 0 ? _mean_observation
                                           : Eigen::VectorXd::Zero(_dim_out);
            }

            const Eigen::MatrixXd& mean_vector() const { return _mean_vector; }

            const Eigen::MatrixXd& obs_mean() const { return _obs_mean; }

            int nb_samples() const { return _samples.size(); }

            int nb_bl_samples() const { return _bl_samples.size(); }

            void update()
            {
                this->_compute_obs_mean(); // ORDER MATTERS
                this->_compute_kernel();
            }

            double get_lik() const { return _lik; }

            void set_lik(const double& lik) { _lik = lik; }

            const Eigen::LLT<Eigen::MatrixXd>& llt() const { return _llt; }

            const Eigen::MatrixXd& alpha() const { return _alpha; }

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
            Eigen::MatrixXd _l_matrix;
            Eigen::LLT<Eigen::MatrixXd> _llt;
            Eigen::MatrixXd _inv_bl_kernel;

            double _lik;

            void _compute_obs_mean()
            {
                _mean_vector.resize(_samples.size(), _dim_out);
                for (int i = 0; i < _mean_vector.rows(); i++)
                    _mean_vector.row(i) = _mean_function(_samples[i], *this);
                _obs_mean = _observations - _mean_vector;
            }

            void _compute_kernel()
            {
                // O(n^2) [should be negligible]
                _kernel.resize(_samples.size(), _samples.size());
                for (size_t i = 0; i < _samples.size(); i++)
                    for (size_t j = 0; j < _samples.size(); ++j)
                        _kernel(i, j) = _kernel_function(_samples[i], _samples[j]) + ((i == j) ? _noise : 0); // noise only on the diagonal

                // O(n^3)
                //  _inverted_kernel = _kernel.inverse();

                _llt = Eigen::LLT<Eigen::MatrixXd>(_kernel);

                // alpha = K^{-1} * this->_obs_mean;
                _alpha = _llt.matrixL().solve(_obs_mean);
                _llt.matrixL().adjoint().solveInPlace(_alpha);
                if (_bl_samples.size() == 0)
                    return;

                Eigen::MatrixXd A1 = Eigen::MatrixXd::Identity(this->_samples.size(), this->_samples.size());
                _llt.matrixL().solveInPlace(A1);
                _llt.matrixL().transpose().solveInPlace(A1);
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
                    _bl_samples.size()) = -A1 * B * comA1;
                _inv_bl_kernel.block(_samples.size(), 0, _bl_samples.size(),
                    _samples.size()) = _inv_bl_kernel.block(0, _samples.size(), _samples.size(),
                                                          _bl_samples.size()).transpose();
                _inv_bl_kernel.block(_samples.size(), _samples.size(), _bl_samples.size(),
                    _bl_samples.size()) = comA1;
            }

            Eigen::VectorXd _mu(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const
            {
                return (k.transpose() * _alpha) + _mean_function(v, *this).transpose();
            }

            double _sigma(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const
            {
                double res;
                if (_bl_samples.size() == 0) {
                    Eigen::VectorXd z = _llt.matrixL().solve(k);
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
