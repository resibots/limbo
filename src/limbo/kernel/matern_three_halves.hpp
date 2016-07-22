#ifndef LIMBO_KERNEL_MATERN_THREE_HALVES_HPP
#define LIMBO_KERNEL_MATERN_THREE_HALVES_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct kernel_maternthreehalves {
            /// @ingroup kernel_defaults
            BO_PARAM(double, sigma_sq, 1);
            /// @ingroup kernel_defaults
            BO_PARAM(double, l, 1);
        };
    }
    namespace kernel {
        /**
         @ingroup kernel
         \rst
         Matern 3/2 kernel

         .. math::
           d = ||v1 - v2||

           \nu = 3/2

           C(d) = \sigma^2\frac{2^{1-\nu}}{\Gamma(\nu)}\Bigg(\sqrt{2\nu}\frac{d}{l}\Bigg)^\nu K_\nu\Bigg(\sqrt{2\nu}\frac{d}{l}\Bigg),


         Parameters:
          - ``double sigma_sq`` (signal variance)
          - ``double l`` (characteristic length scale)

        Reference: :cite:`matern1960spatial` & :cite:`brochu2010tutorial` p.10 & https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function
        \endrst
        */
        template <typename Params>
        struct MaternThreeHalves {
            MaternThreeHalves(size_t dim = 1) {}

            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double d = (v1 - v2).norm();
                return Params::kernel_maternthreehalves::sigma_sq() * (1 + std::sqrt(3) * d / Params::kernel_maternthreehalves::l()) * std::exp(-std::sqrt(3) * d / Params::kernel_maternthreehalves::l());
            }
        };
    }
}

#endif
