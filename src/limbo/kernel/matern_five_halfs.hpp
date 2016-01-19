#ifndef LIMBO_KERNEL_MATERN_FIVE_HALFS_HPP
#define LIMBO_KERNEL_MATERN_FIVE_HALFS_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct kernel_maternfivehalfs {
            BO_PARAM(double, sigma, 1);
            BO_PARAM(double, l, 1);
        };
    }
    namespace kernel {
        template <typename Params>
        struct MaternFiveHalfs {
            MaternFiveHalfs(size_t dim = 1) {}

            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double d = (v1 - v2).norm();
                return Params::kernel_maternfivehalfs::sigma() * (1 + std::sqrt(5) * d / Params::kernel_maternfivehalfs::l() + 5 * d * d / (3 * Params::kernel_maternfivehalfs::l() * Params::kernel_maternfivehalfs::l())) * std::exp(-std::sqrt(5) * d / Params::kernel_maternfivehalfs::l());
            }
        };
    }
}

#endif
