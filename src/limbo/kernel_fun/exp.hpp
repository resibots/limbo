#ifndef KERNEL_FUNCTIONS_EXP_HPP_
#define KERNEL_FUNCTIONS_EXP_HPP_

#include <Eigen/Core>

namespace limbo {
    namespace kernel_fun {
        template <typename Params>
        struct Exp {
            Exp(size_t dim = 1) {}
            double operator()(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) const
            {
                double _sigma = Params::kf_exp::sigma();
                return (exp(-(1 / (2 * pow(_sigma, 2))) * pow((v1 - v2).norm(), 2)));
            }
        };
    }
}

#endif