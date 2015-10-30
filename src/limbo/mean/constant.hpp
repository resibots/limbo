#ifndef LIMBO_MEAN_CONSTANT_HPP
#define LIMBO_MEAN_CONSTANT_HPP

#include <Eigen/Core>

namespace limbo {
    namespace mean {
        template <typename Params>
        struct Constant {
            Constant(size_t dim_out = 1) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP&) const
            {
                return Params::meanconstant::constant();
            }
        };
    }
}

#endif
