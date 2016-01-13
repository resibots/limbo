#ifndef LIMBO_ACQUI_GP_UCB_HPP
#define LIMBO_ACQUI_GP_UCB_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct acqui_gpucb {
            BO_PARAM(double, delta, 0.001);
        };
    }
    namespace acqui {
        template <typename Params, typename Model>
        class GP_UCB {
        public:
            GP_UCB(const Model& model, int iteration) : _model(model)
            {
                double t3 = std::pow(iteration, 3.0);
                static constexpr double delta3 = Params::acqui_gpucb::delta() * 3;
                static constexpr double pi2 = M_PI * M_PI;
                _beta = std::sqrt(2.0 * std::log(t3 * pi2 / delta3));
            }

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            template <typename AggregatorFunction>
            double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
            {
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query(v);
                return (afun(mu) + _beta * std::sqrt(sigma));
            }

        protected:
            const Model& _model;
            double _beta;
        };
    }
}
#endif
