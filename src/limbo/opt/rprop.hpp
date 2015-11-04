#ifndef LIMBO_OPT_RPROP_HPP
#define LIMBO_OPT_RPROP_HPP

#include <algorithm>

#include <boost/math/special_functions/sign.hpp>

#include <Eigen/Core>

#include <limbo/tools/parallel.hpp>

namespace limbo {
    namespace defaults {
        struct rprop {
            BO_PARAM(int, iters, 300);
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
            Eigen::VectorXd operator()(F& f)
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
                Eigen::VectorXd best_params = params;
                double best = log(0);

                for (int i = 0; i < Params::rprop::iters(); ++i) {
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
