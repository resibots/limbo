#ifndef LIMBO_KERNEL_MATERN_THREE_HALFS_HPP
#define LIMBO_KERNEL_MATERN_THREE_HALFS_HPP

#include <Eigen/Core>

namespace limbo {
    namespace kernel {
        template <typename Params>
        struct MaternThreeHalfs {
            MaternThreeHalfs(size_t dim = 1) {}

            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double d = (v1 - v2).norm();
                return Params::kf_maternthreehalfs::sigma() * (1 + sqrt(3) * d / Params::kf_maternthreehalfs::l()) * exp(-sqrt(3) * d / Params::kf_maternthreehalfs::l());
            }
        };
    }
}

#endif
