#ifndef LIMBO_ACQUI_GP_UCB_HPP
#define LIMBO_ACQUI_GP_UCB_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {

    namespace defaults {
        struct gp_ucb {
            BO_PARAM(double, delta, 0.001);
        };
    }

    namespace acqui {
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

            template <typename AggregatorFunction>
            double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
            {
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query(v);
                return (afun(mu) + _beta * sqrt(sigma));
            }

        protected:
            const Model& _model;
            double _beta;
        };
    }
}
#endif
