#ifndef ACQUISITION_FUNCTIONS_HPP_
#define ACQUISITION_FUNCTIONS_HPP_

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <limits>

namespace limbo {

    namespace defaults {
        struct gp_ucb {
            BO_PARAM(float, delta, 0.001);
        };

        struct ucb {
            BO_PARAM(float, alpha, 0.5);
        };
    }

    namespace acquisition_functions {
        template <typename Params, typename Model>
        class UCB {
        public:
            UCB(const Model& model, int iteration = 0) : _model(model) {}

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            template <typename RewardFunction>
            double operator()(const Eigen::VectorXd& v, const RewardFunction& rfun) const
            {
                // double mu, sigma;
                // std::tie(mu, sigma) = _model.query(v);
                // return (mu + Params::ucb::alpha() * sqrt(sigma));
                return (rfun(_model.mu(v)) + Params::ucb::alpha() * sqrt(_model.sigma(v)));
            }

        protected:
            const Model& _model;
        };

        template <typename Params, typename Model>
        class GP_UCB {
        public:
            GP_UCB(const Model& model, int iteration) : _model(model)
            {
                double t3 = pow(iteration, 3.0);
                static constexpr double delta3 = Params::gp_ucb::delta() * 3;
                static constexpr double pi2 = M_PI * M_PI;
                _beta = sqrtf(2.0 * log(t3 * pi2 / delta3));
            }

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            template <typename RewardFunction>
            double operator()(const Eigen::VectorXd& v, const RewardFunction& rfun) const
            {
                // double mu, sigma;
                // std::tie(mu, sigma) = _model.query(v);
                // return (mu + _beta * sqrt(sigma));
                return (rfun(_model.mu(v)) + _beta * sqrt(_model.sigma(v)));
            }

        protected:
            const Model& _model;
            double _beta;
        };
    }
}
#endif
