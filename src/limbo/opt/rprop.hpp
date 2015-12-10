#ifndef LIMBO_OPT_RPROP_HPP
#define LIMBO_OPT_RPROP_HPP

#include <algorithm>

#include <boost/math/special_functions/sign.hpp>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
   namespace defaults {
        struct opt_rprop {
            BO_PARAM(int, iterations, 300);
        };
    }
    namespace opt {
        // partly inspired by libgp: https://github.com/mblum/libgp
        // reference :
        // Blum, M., & Riedmiller, M. (2013). Optimization of Gaussian
        // Process Hyperparameters using Rprop. In European Symposium
        // on Artificial Neural Networks, Computational Intelligence
        // and Machine Learning.
        template <typename Params>
        struct Rprop {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, bool bounded) const
            {
                // params
                size_t param_dim = f.param_size();
                double delta0 = 0.1;
                double deltamin = 1e-6;
                double deltamax = 50;
                double etaminus = 0.5;
                double etaplus = 1.2;
                double eps_stop = 0.0;

                Eigen::VectorXd delta = Eigen::VectorXd::Ones(param_dim) * delta0;
                Eigen::VectorXd grad_old = Eigen::VectorXd::Zero(param_dim);
                Eigen::VectorXd params = f.init();

                if (bounded) {
                    for (int j = 0; j < params.size(); j++) {
                        if (params(j) < 0)
                            params(j) = 0;
                        if (params(j) > 1)
                            params(j) = 1;
                    }
                }

                Eigen::VectorXd best_params = params;
                double best = log(0);

                for (int i = 0; i < Params::opt_rprop::iterations(); ++i) {
                    auto perf = f.utility_and_grad(params);
                    double lik = std::get<0>(perf);
                    if (lik > best) {
                        best = lik;
                        best_params = params;
                    }
                    Eigen::VectorXd grad = -std::get<1>(perf);
                    grad_old = grad_old.cwiseProduct(grad);

                    for (int j = 0; j < grad_old.size(); ++j) {
                        if (grad_old(j) > 0) {
                            delta(j) = std::min(delta(j) * etaplus, deltamax);
                        }
                        else if (grad_old(j) < 0) {
                            delta(j) = std::max(delta(j) * etaminus, deltamin);
                            grad(j) = 0;
                        }
                        params(j) += -boost::math::sign(grad(j)) * delta(j);

                        if (bounded && params(j) < 0)
                            params(j) = 0;
                        if (bounded && params(j) > 1)
                            params(j) = 1;
                    }

                    grad_old = grad;
                    if (grad_old.norm() < eps_stop)
                        break;
                }

                return best_params;
            }
        };
    }
}

#endif
