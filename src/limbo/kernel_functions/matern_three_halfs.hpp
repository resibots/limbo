#ifndef KERNEL_FUNCTIONS_MATERN_HPP_
#define KERNEL_FUNCTIONS_MATERN_HPP_

#include <Eigen/Core>

namespace limbo {
    namespace kernel_functions {
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