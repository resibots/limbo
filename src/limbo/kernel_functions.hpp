#ifndef KERNEL_FUNCTIONS_HPP_
#define KERNEL_FUNCTIONS_HPP_




#include <Eigen/Core>

namespace limbo {
  namespace kernel_functions {
    template<typename Params>
    struct Exp {
      Exp(size_t dim = 1) { }
      double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const {
        double _sigma = Params::kf_exp::sigma;
        return (exp(-(1 / (2 * pow(_sigma, 2))) * pow((v1 - v2).norm(), 2)));
      }
    };

    template<typename Params>
    struct MaternThreeHalfs {
      MaternThreeHalfs(size_t dim = 1) {
      }

      double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const {
        double d = (v1 - v2).norm();
        return  Params::kf_maternthreehalfs::sigma()
                * (1 + sqrt(3) * d / Params::kf_maternthreehalfs::l())
                * exp(-sqrt(3) * d / Params::kf_maternthreehalfs::l());
      }
    };

    template<typename Params>
    struct MaternFiveHalfs {
      MaternFiveHalfs(size_t dim = 1) {
      }

      double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const {
        double d = (v1 - v2).norm();
        return  Params::kf_maternfivehalfs::sigma()
                * (1 + sqrt(5) * d / Params::kf_maternfivehalfs::l()
                   + 5 * d * d / (3 * Params::kf_maternfivehalfs::l()
                                  * Params::kf_maternfivehalfs::l())) * exp(-sqrt(5) * d
                                      / Params::kf_maternfivehalfs::l());
      }
    };

    /** Squared exponential covariance function with automatic relevance detection.
    * Computes the squared exponential covariance
    * \f$k_{SE}(x, y) := \alpha^2 \exp(-\frac{1}{2}(x-y)^T\Lambda^{-1}(x-y))\f$,
    * with \f$\Lambda = diag(l_1^2, \dots, l_n^2)\f$ being the characteristic
    * length scales and \f$\alpha\f$ describing the variability of the latent
    * function. The parameters \f$l_1^2, \dots, l_n^2, \alpha\f$ are expected
    * in this order in the parameter array.
    */
    template<typename Params>
    struct SquaredExpARD {
      SquaredExpARD(int dim) : _sf2(0), _ell(dim), _input_dim(dim) {
        this->set_h_params(Eigen::VectorXd::Ones(_ell.size() + 1) * -1);
      }
      size_t h_params_size() const {
        return _ell.size() + 1;
      }
      const Eigen::VectorXd& h_params() const {
        return _h_params;
      }
      void set_h_params(const Eigen::VectorXd& p) {
        _h_params = p;
        for (size_t i = 0; i < _input_dim; ++i)
          _ell(i) = exp(p(i));
        _sf2 = exp(2 * p(_input_dim));
      }

      Eigen::VectorXd grad(const Eigen::VectorXd &x1, const Eigen::VectorXd &x2) const {
        Eigen::VectorXd grad(_input_dim + 1);
        Eigen::VectorXd z = (x1 - x2).cwiseQuotient(_ell).array().square();
        double k = _sf2 * exp(-0.5 * z.sum());
        grad.head(_input_dim) = z * k;
        grad(_input_dim) = 2.0 * k;
        return grad;
      }

      double operator()(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2) const {
        assert(x1.size() == _ell.size());
        double z = (x1 - x2).cwiseQuotient(_ell).squaredNorm();
        return _sf2 * exp(-0.5 * z);
      }
      const Eigen::VectorXd& ell() const {
        return _ell;
      }
     protected:
      double _sf2;
      Eigen::VectorXd _ell;
      size_t _input_dim;
      Eigen::VectorXd _h_params;

    };
  }
}
#endif
