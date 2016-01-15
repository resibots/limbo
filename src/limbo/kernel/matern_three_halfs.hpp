#ifndef LIMBO_KERNEL_MATERN_THREE_HALFS_HPP
#define LIMBO_KERNEL_MATERN_THREE_HALFS_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct kernel_maternthreehalfs {
            BO_PARAM(double, sigma, 1);
            BO_PARAM(double, l, 1);
        };
    }
    namespace kernel {
        template <typename Params>
        struct MaternThreeHalfs {
            MaternThreeHalfs(size_t dim = 1) {}

            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double d = (v1 - v2).norm();
                return Params::kernel_maternthreehalfs::sigma() * (1 + std::sqrt(3) * d / Params::kernel_maternthreehalfs::l()) * std::exp(-std::sqrt(3) * d / Params::kernel_maternthreehalfs::l());
            }
        };
    }
}

#endif
