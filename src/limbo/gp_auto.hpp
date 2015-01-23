#ifndef GP_AUTO_HPP_
#define GP_AUTO_HPP_

#include <limits>
#include <cassert>
#include <tbb/parallel_reduce.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/LU>

#include "gp.hpp"
#include "rprop.hpp"
#include "parallel.hpp"

namespace limbo {
  namespace defaults {
    struct gp_auto {
      BO_PARAM(int, n_rprop, 100);
      BO_PARAM(int, rprop_restart, 100);
    };
  }
  namespace model {
    template<typename Params, typename KernelFunction, typename MeanFunction>
    class GPAuto : public GP<Params, KernelFunction, MeanFunction> {
     public:
      // TODO : init KernelFunction with dim in GP
      GPAuto(int d) : GP<Params, KernelFunction, MeanFunction>(d) {}

      void compute(const std::vector<Eigen::VectorXd>& samples,
                   const std::vector<double>& observations,
                   double noise) {

        GP<Params, KernelFunction, MeanFunction>::compute(samples, observations, noise);
        _optimize_likelihood();
        this->_compute_kernel();
      }

      // see Rasmussen and Williams, 2006 (p. 113)
      double log_likelihood(const Eigen::VectorXd& h_params,
                            bool update_kernel = true) {
        this->_kernel_function.set_h_params(h_params);
        if (update_kernel)
          this->_compute_kernel();
        size_t n = this->_obs_mean.size();

        // --- cholesky ---
        // see: http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
        Eigen::MatrixXd l = this->_llt.matrixL();
        long double det = 2 * l.diagonal().array().log().sum();

        // alpha = K^{-1} * this->_obs_mean;
        double a = this->_obs_mean.dot(this->_alpha);

        return -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);
      }

      // see Rasmussen and Williams, 2006 (p. 114)
      Eigen::VectorXd log_likelihood_grad(const Eigen::VectorXd& h_params,
                                          bool update_kernel = true) {
        this->_kernel_function.set_h_params(h_params);
        if (update_kernel)
          this->_compute_kernel();
        size_t n = this->_observations.size();

        /// what we should write, but it is less numerically stable than using the Cholesky decomposition
        // Eigen::MatrixXd alpha = this->_inverted_kernel * this->_obs_mean;
        //  Eigen::MatrixXd w = alpha * alpha.transpose() - this->_inverted_kernel;

        // alpha = K^{-1} * this->_obs_mean;
        Eigen::VectorXd alpha = this->_llt.matrixL().solve(this->_obs_mean);
        this->_llt.matrixL().adjoint().solveInPlace(alpha);

        // K^{-1} using Cholesky decomposition
        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);
        this->_llt.matrixL().solveInPlace(w);
        this->_llt.matrixL().transpose().solveInPlace(w);

        // alpha * alpha.transpose() - K^{-1}
        w = alpha * alpha.transpose() - w;

        // only compute half of the matrix (symmetrical matrix)
        Eigen::VectorXd grad =
          Eigen::VectorXd::Zero(this->_kernel_function.h_params_size());
        for (size_t i = 0; i < n; ++i) {
          for (size_t j = 0; j <= i; ++j) {
            Eigen::VectorXd g = this->_kernel_function.grad(this->_samples[i], this->_samples[j]);
            if (i == j)
              grad += w(i, j) * g * 0.5;
            else
              grad += w(i, j) * g;
          }
        }
        return grad;
      }
     protected:
      void _optimize_likelihood() {
        par::init();
        typedef std::pair<Eigen::VectorXd, double> pair_t;
        auto body = [ = ](int i) {
          // we need a copy because each thread should touch a copy of the GP!
          auto gp = *this;
          Eigen::VectorXd v = rprop::optimize([&](const Eigen::VectorXd & v) {
            return gp.log_likelihood(v);
          },
          [&](const Eigen::VectorXd & v) {
            return gp.log_likelihood_grad(v, false);
          },
          this->kernel_function().h_params_size(), Params::gp_auto::n_rprop());
          double lik = gp.log_likelihood(v);
          return std::make_pair(v, lik);
        };
        auto comp = [](const pair_t& v1, const pair_t& v2) {
          return v1.second > v2.second;
        };
        pair_t init(Eigen::VectorXd::Zero(1), -std::numeric_limits<float>::max());
        auto m = par::max(init, Params::gp_auto::rprop_restart(), body, comp);
        std::cout << "likelihood:" << m.second << std::endl;
        this->_kernel_function.set_h_params(m.first);

      }
    };
  }
}
#endif
