#ifndef GP_HPP_
#define GP_HPP_

#include <cassert>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>


#include "kernel_functions.hpp"
#include "mean_functions.hpp"

namespace limbo {
  namespace model {
    template<typename Params, typename KernelFunction, typename MeanFunction>
    class GP {
     public:
      GP() : _dim(-1) {}
      // useful because the model might created  before having samples
      GP(int d) : _dim(d), _kernel_function(d) {}

      void compute(const std::vector<Eigen::VectorXd>& samples,
                   const std::vector<double>& observations,
                   double noise) {
        if (_dim == -1) {
          assert(samples.size() != 0);
          assert(observations.size() != 0);
          assert(samples.size() == observations.size());
          _dim = samples[0].size();
        }
        _samples = samples;
        _observations.resize(observations.size());
        _noise = noise;

        for (int i = 0; i < _observations.size(); ++i)
          _observations(i) = observations[i];
        _mean_observation = _observations.sum() / _observations.size();

        _mean_vector.resize(_samples.size());
        for (int i = 0; i < _mean_vector.size(); i++)
          _mean_vector(i) = _mean_function(_samples[i], *this);
        _obs_mean = _observations - _mean_vector;

        _compute_kernel();
      }

      // return mu, sigma (unormaliz)
      std::tuple<double, double> query(const Eigen::VectorXd& v) const {
        if (_samples.size() == 0)
          return std::make_tuple(_mean_function(v, *this),
                                 sqrt(_kernel_function(v, v)));

        Eigen::VectorXd k = _compute_k(v);
        return std::make_tuple(_mu(v, k), _sigma(v, k));
      }

      double mu(const Eigen::VectorXd& v) const {
        if (_samples.size() == 0)
          return _mean_function(v, *this);
        return _mu(v, _compute_k(v));
      }
      double sigma(const Eigen::VectorXd& v) const {
        if (_samples.size() == 0)
          return sqrt(_kernel_function(v, v));
        return _sigma(v, _compute_k(v));
      }
      int dim() const {
        assert(_dim != -1);//need to compute first !
        return _dim;
      }
      const KernelFunction& kernel_function() const {
        return _kernel_function;
      }
      const MeanFunction& mean_function() const {
        return _mean_function;
      }
      double max_observation() const {
        return _observations.maxCoeff();
      }
      double mean_observation() const {
        return _mean_observation;
      }
     protected:
      int _dim;
      KernelFunction _kernel_function;
      MeanFunction _mean_function;

      std::vector<Eigen::VectorXd> _samples;
      Eigen::VectorXd _observations;
      Eigen::VectorXd _mean_vector;
      Eigen::VectorXd _obs_mean;

      double _noise;
      Eigen::VectorXd _alpha;
      double _mean_observation;

      Eigen::MatrixXd _kernel;
      Eigen::MatrixXd _inverted_kernel;
      Eigen::MatrixXd _l_matrix;
      Eigen::LLT<Eigen::MatrixXd> _llt;

      void _compute_kernel() {
        // O(n^2) [should be negligible]
        _kernel.resize(_observations.size(), _observations.size());
        for (int i = 0; i < _observations.size(); i++)
          for (int j = 0; j < _observations.size(); ++j)
            _kernel(i, j) = _kernel_function(_samples[i], _samples[j])
                            +  ( (i == j) ? _noise : 0); // noise only on the diagonal

        // O(n^3)
        //  _inverted_kernel = _kernel.inverse();

        _llt = Eigen::LLT<Eigen::MatrixXd>(this->_kernel);

        // alpha = K^{-1} * this->_obs_mean;
        _alpha = _llt.matrixL().solve(this->_obs_mean);
        _llt.matrixL().adjoint().solveInPlace(_alpha);
      }

      double _mu(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const {
        return _mean_function(v, *this) + k.transpose() * _alpha;
        //        return _mean_function(v)
        //               + (k.transpose() * _inverted_kernel * (_obs_mean))[0];
      }
      double _sigma(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const {
        Eigen::VectorXd z = _llt.matrixL().solve(k);
        return  _kernel_function(v, v) - z.dot(z);
        //        return  _kernel_function(v, v) - (k.transpose() * _inverted_kernel * k)[0];
      }
      Eigen::VectorXd _compute_k(const Eigen::VectorXd& v) const {
        Eigen::VectorXd k(_samples.size());
        for (int i = 0; i < k.size(); i++)
          k[i] = _kernel_function(_samples[i], v);
        return k;
      }
    };
  }
}
#endif
