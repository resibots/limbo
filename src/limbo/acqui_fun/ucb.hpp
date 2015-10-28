#ifndef ACQUISITION_FUNCTIONS_UCB_HPP_
#define ACQUISITION_FUNCTIONS_UCB_HPP_

#include <Eigen/Core>

namespace limbo {

    namespace defaults {
        struct ucb {
            BO_PARAM(float, alpha, 0.5);
        };
    }

    namespace acqui_fun {
        template <typename Params, typename Model>
        class UCB {
        public:
            UCB(const Model& model, int iteration = 0) : _model(model) {}

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            template <typename AggregatorFunction>
            double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
            {
                Eigen::VectorXd mu;
                double sigma;
                std::tie(mu, sigma) = _model.query(v);
                return (afun(mu) + Params::ucb::alpha() * sqrt(sigma));
            }

        protected:
            const Model& _model;
        };
    }
}

#endif