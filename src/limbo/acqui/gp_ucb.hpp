#ifndef LIMBO_ACQUI_GP_UCB_HPP
#define LIMBO_ACQUI_GP_UCB_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct acqui_gpucb {
            /// @ingroup acqui_defaults
            BO_PARAM(double, delta, 0.1);
        };
    }
    namespace acqui {
        /** @ingroup acqui
        \rst
        GP-UCB (Upper Confidence Bound). See :cite:`brochu2010tutorial`, p. 15. See also: http://arxiv.org/abs/0912.3995

        .. math::
          UCB(x) = \mu(x) + \kappa \sigma(x).

        with:

        .. math::
          \kappa = \sqrt{2 \log{(\frac{n^{D/2+2}\pi^2}{3 \delta})}}

        where :math:`n` is the number of past evaluations of the objective function and :math:`D` the dimensionality of the parameters (dim_in).

        Parameters:
          - `double delta` (a small number in [0,1], e.g. 0.1)
        \endrst
        */
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
