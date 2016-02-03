#ifndef LIMBO_ACQUI_UCB_HPP
#define LIMBO_ACQUI_UCB_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct acqui_ucb_imgpo {
            BO_PARAM(double, nu, 0.05);
        };
    }
    namespace acqui {
        namespace experimental {
            template <typename Params, typename Model>
            class UCB_IMGPO {
            public:
                UCB_IMGPO(const Model& model, size_t M = 1) : _model(model), _M(M) {}

                size_t dim_in() const { return _model.dim_in(); }

                size_t dim_out() const { return _model.dim_out(); }

                template <typename AggregatorFunction>
                double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
                {
                    Eigen::VectorXd mu;
                    double sigma;
                    std::tie(mu, sigma) = _model.query(v);
                    // UCB - nu = 0.05
                    // sqrt(2*log(pi^2*M^2/(12*nu)))
                    double gp_varsigma = std::sqrt(2.0 * std::log(std::pow(M_PI, 2.0) * std::pow(_M, 2.0) / (12.0 * Params::acqui_ucb_imgpo::nu())));
                    return (afun(mu) + (gp_varsigma + 0.2) * std::sqrt(sigma));
                }

            protected:
                const Model& _model;
                size_t _M;
            };
        }
    }
}

#endif
