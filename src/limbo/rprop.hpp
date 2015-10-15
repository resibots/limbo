#ifndef RPROP_HPP_
#define RPROP_HPP_

#include <Eigen/Core>
#include <boost/math/special_functions/sign.hpp>


namespace rprop {

  // partly inspired by libgp: https://github.com/mblum/libgp
  // reference :
  // Blum, M., & Riedmiller, M. (2013). Optimization of Gaussian
  // Process Hyperparameters using Rprop. In European Symposium
  // on Artificial Neural Networks, Computational Intelligence
  // and Machine Learning.
  template<typename F1, typename F2>
  Eigen::VectorXd optimize(const F1& func, const F2& grad_func, int param_dim, int n) {
    // params
    double delta0 = 0.1;
    double deltamin = 1e-6;
    double deltamax = 50;
    double etaminus = 0.5;
    double etaplus = 1.2;
    double eps_stop = 0.0;

    Eigen::VectorXd delta = Eigen::VectorXd::Ones(param_dim) * delta0;
    Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
    Eigen::VectorXd params = Eigen::VectorXd::Ones(param_dim);
//(Eigen::VectorXd::Random(param_dim).array() - 1);
    Eigen::VectorXd best_params = params;
    double best = log(0);

    for (int i = 0; i < n; ++i) {
      double lik = func(params);
      if (lik > best) {
        best = lik;
        best_params = params;
      }
      Eigen::VectorXd grad = -grad_func(params);
      grad_old = grad_old.cwiseProduct(grad);
      for (int j = 0; j < grad_old.size(); ++j) {
        if (grad_old(j) > 0) {
          delta(j) = std::min(delta(j) * etaplus, deltamax);
        } else if (grad_old(j) < 0) {
          delta(j) = std::max(delta(j) * etaminus, deltamin);
          grad(j) = 0;
        }
        params(j) += -boost::math::sign(grad(j)) * delta(j);
      }
      grad_old = grad;
      if (grad_old.norm() < eps_stop) break;

    
    }
    return best_params;
  }

}
#endif
