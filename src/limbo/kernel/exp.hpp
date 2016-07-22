#ifndef LIMBO_KERNEL_EXP_HPP
#define LIMBO_KERNEL_EXP_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct kernel_exp {
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 1);
        };
    }
    namespace kernel {
        /**
          @ingroup kernel
          \rst
          Exponential kernel (see :cite:`brochu2010tutorial` p. 9).

          .. math::
              k(v_1, v_2)  = \sigma^2\exp \Big(-\frac{1}{l^2} ||v_1 - v_2||^2\Big)

          Parameters:
            - ``double sigma_sq`` (signal variance)
            - ``double l`` (characteristic length scale)
          \endrst
        */
        template <typename Params>
        struct Exp {
            Exp(size_t dim = 1) {}
            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double _l = Params::kernel_exp::l();
                return Params::kernel_exp::sigma_sq() * (std::exp(-(1 / (2 * std::pow(_l, 2))) * std::pow((v1 - v2).norm(), 2)));
            }
        };
    }
}

#endif
