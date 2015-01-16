
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


    template<typename Params, typename KernelFunction, typename MeanFunction, typename ObsType= Eigen::VectorXd>
    class GP {
     public:
      GP() : _dim(-1) {}
      // useful because the model might be created  before having samples
      GP(int d) : _dim(d), _kernel_function(d) {}

      void compute(const std::vector<Eigen::VectorXd>& samples,
                   const std::vector<ObsType>& observations,
                   double noise) {

        if (_dim == -1) {
          assert(samples.size() != 0);
          assert(observations.size() != 0);
          assert(samples.size() == observations.size());
          _dim = samples[0].size();
        }
        _samples = samples;
        _observations.resize(observations.size(),observations[0].size());
        _noise = noise;

	int obs_dim=observations[0].size();

        for (int i = 0; i < _observations.rows(); ++i) 
          _observations.row(i) = observations[i];
	_mean_observation.resize(obs_dim);
	for(int i=0; i< _observations.cols(); i++)
	  _mean_observation(i) = _observations.col(i).sum() / _observations.rows();

        _mean_vector.resize(_samples.size(),obs_dim);
        for (int i = 0; i < _mean_vector.rows(); i++)
	  _mean_vector.row(i) = ObsType::Zero(obs_dim)+_mean_function(_samples[i], *this); // small trick to accept either Double or Vector
        _obs_mean = _observations - _mean_vector;


        _compute_kernel();

      }

      // return mu, sigma (unormaliz)
      std::tuple<ObsType, double> query(const Eigen::VectorXd& v) const {
        if (_samples.size() == 0)
          return std::make_tuple(_mean_function(v, *this),
                                 sqrt(_kernel_function(v, v)));

        Eigen::VectorXd k = _compute_k(v);
        return std::make_tuple(_mu(v, k), _sigma(v, k));
      }

      ObsType mu(const Eigen::VectorXd& v) const {
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
      ObsType max_observation() const {  
	if(_observations.cols()>1)
	  std::cout<<"WARNING max_observation with multi dimensional observations doesn't make sense"<<std::endl;
        return _observations.maxCoeff();
      }
      ObsType mean_observation() const {
        return _mean_observation;
      }
     protected:
      int _dim;
      KernelFunction _kernel_function;
      MeanFunction _mean_function;

      std::vector<Eigen::VectorXd> _samples;
      Eigen::MatrixXd _observations;
      Eigen::MatrixXd _mean_vector;
      Eigen::MatrixXd _obs_mean;

      double _noise;
      Eigen::MatrixXd _alpha;
      ObsType _mean_observation;

      Eigen::MatrixXd _kernel;
      Eigen::MatrixXd _inverted_kernel;
      Eigen::MatrixXd _l_matrix;
      Eigen::LLT<Eigen::MatrixXd> _llt;

      void _compute_kernel() {
        // O(n^2) [should be negligible]
        _kernel.resize(_samples.size(), _samples.size());
        for (size_t i = 0; i < _samples.size(); i++)
          for (size_t j = 0; j < _samples.size(); ++j)
            _kernel(i, j) = _kernel_function(_samples[i], _samples[j]) + _noise;

        // O(n^3)
        //  _inverted_kernel = _kernel.inverse();

        _llt = Eigen::LLT<Eigen::MatrixXd>(this->_kernel);

        // alpha = K^{-1} * this->_obs_mean;
        _alpha = _llt.matrixL().solve(this->_obs_mean);
        _llt.matrixL().adjoint().solveInPlace(_alpha);

      }

      ObsType _mu(const Eigen::VectorXd& v, const Eigen::VectorXd& k) const {
        return  (k.transpose() * _alpha) + _mean_function(v, *this);
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
