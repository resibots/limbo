#ifndef KERNEL_FUNCTIONS_MATERN_FIVE_HALFS_HPP_
#define KERNEL_FUNCTIONS_MATERN_FIVE_HALFS_HPP_

#include <Eigen/Core>

namespace limbo {
    namespace kernel_fun {
        template <typename Params>
        struct MaternFiveHalfs {
            MaternFiveHalfs(size_t dim = 1) {}

            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double d = (v1 - v2).norm();
                return Params::kf_maternfivehalfs::sigma() * (1 + sqrt(5) * d / Params::kf_maternfivehalfs::l() + 5 * d * d / (3 * Params::kf_maternfivehalfs::l() * Params::kf_maternfivehalfs::l())) * exp(-sqrt(5) * d / Params::kf_maternfivehalfs::l());
            }
        };
    }
}

#endif