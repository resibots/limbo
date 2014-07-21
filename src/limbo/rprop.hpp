#ifndef RPROP_HPP_
#define RPROP_HPP_

#include <Eigen/Core>
#include <boost/math/special_functions/sign.hpp>
namespace rprop {

  template<typename F1, typename F2>
  Eigen::VectorXd optimize(const F1& func, const F2& grad_func, int param_dim, int n) {
    // params
    double Delta0 = 0.1;
    double Deltamin = 1e-6;
    double Deltamax = 50;
    double etaminus = 0.5;
    double etaplus = 1.2;
    double eps_stop = 0.0;

    Eigen::VectorXd Delta = Eigen::VectorXd::Ones(param_dim) * Delta0;
    Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
    Eigen::VectorXd params = (Eigen::VectorXd::Random(param_dim).array() + 1); //0.5
    //  Eigen::VectorXd::Constant(param_dim, 1);
    Eigen::VectorXd best_params = params;
    double best = log(0);

    for (size_t i = 0; i < n; ++i) {
      double lik = func(params);
      Eigen::VectorXd grad = -grad_func(params);
      grad_old = grad_old.cwiseProduct(grad);
      for (int j = 0; j < grad_old.size(); ++j) {
        if (grad_old(j) > 0) {
          Delta(j) = std::min(Delta(j) * etaplus, Deltamax);
        } else if (grad_old(j) < 0) {
          Delta(j) = std::max(Delta(j) * etaminus, Deltamin);
          grad(j) = 0;
        }
        params(j) += -boost::math::sign(grad(j)) * Delta(j);
      }
      grad_old = grad;
      if (grad_old.norm() < eps_stop) break;

      if (lik > best) {
        best = lik;
        best_params = params;
      }
    }
    //  std::cout << "rprop: " << -best << std::endl;

    return best_params;
  }

}
#endif
