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
      BO_PARAM(int, rprop_restart, 20);
    };
  }
  namespace model {
    namespace impl {
#ifdef USE_TBB
      template<typename GP>
      struct RpropOpt {
        RpropOpt(const GP& gp, int n_rprop) : _gp(gp), _n_rprop(n_rprop) {}
        void operator()(const tbb::blocked_range<size_t>& r) {
          size_t end = r.end();
          for ( size_t i = r.begin(); i != end; ++i ) {
            Eigen::VectorXd v = rprop::optimize([&](const Eigen::VectorXd & v) {
              return _gp.log_likelihood(v);
            },
            [&](const Eigen::VectorXd & v) {
              return _gp.log_likelihood_grad(v, false);
            },
            _gp.kernel_function().h_params_size(), _n_rprop);
            double lik = _gp.log_likelihood(v);
            _keep_best(lik, v);
          }
        }
        RpropOpt(RpropOpt& x, tbb::split) :
          _gp(x._gp),
          _best_score(x._best_score),
          _best_v(x._best_v) {}
        void join(const RpropOpt& y) {
          _keep_best(y._best_score, y._best_v);
        }
        const Eigen::VectorXd& best_v() const { return _best_v; }
        double best_score() const { return _best_score; }
       protected:
        void _keep_best(float y, const Eigen::VectorXd& v) {
          if (y > _best_score) {
            _best_score = y;
            _best_v = v;
          }
        }
        GP _gp;
        double _best_score = -std::numeric_limits<float>::max();
        Eigen::VectorXd _best_v;
        int _n_rprop;
      };
    }
#endif

    template<typename Params, typename KernelFunction, typename MeanFunction>
    class GPAuto : public GP<Params, KernelFunction, MeanFunction> {
     public:
      // TODO : init KernelFunction with dim in GP
      GPAuto(int d) : GP<Params, KernelFunction, MeanFunction>(d) {}

      void compute(const std::vector<Eigen::VectorXd>& samples,
                   const std::vector<double>& observations,
                   double noise) {

        GP<Params, KernelFunction, MeanFunction>::compute(samples, observations, noise);
#ifdef USE_TBB
        _optimize_likelihood_tbb();
#else
        _optimize_likelihood();
#endif
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
        Eigen::VectorXd alpha = this->_llt.solve(this->_obs_mean);
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
              grad += w(i, j) * g * 0.5; // why 0.5?
            else
              grad += w(i, j) * g;
          }
        }

        return grad;
      }
     protected:
#ifdef USE_TBB
       void _optimize_likelihood_tbb() {
         par::init();
          impl::RpropOpt<GPAuto> r(*this, Params::gp_auto::n_rprop());
          tbb::parallel_reduce(tbb::blocked_range<size_t>(0, Params::gp_auto::rprop_restart()), r);
          this->_kernel_function.set_h_params(r.best_v());
       }
#endif
      void _optimize_likelihood() {
        double best_score = -std::numeric_limits<float>::max();
        Eigen::VectorXd best_v;
        for (size_t i = 0; i < Params::gp_auto::rprop_restart(); ++i) {
          Eigen::VectorXd v = rprop::optimize([&](const Eigen::VectorXd & v) {
            return log_likelihood(v);
          },
          [&](const Eigen::VectorXd & v) {
            return log_likelihood_grad(v, false);
          },
          this->_kernel_function.h_params_size(), Params::gp_auto::n_rprop());
          double lik = log_likelihood(v);
          if (lik > best_score) {
            best_score = lik;
            best_v = v;
          }
        }
        std::cout << "Best lik:" << best_score << std::endl;
        this->_kernel_function.set_h_params(best_v);
      }
    };
  }
}
#endif
