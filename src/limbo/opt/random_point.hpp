#ifndef LIMBO_OPT_RANDOM_POINT_HPP
#define LIMBO_OPT_RANDOM_POINT_HPP

#include <Eigen/Core>

namespace limbo {
    namespace opt {
        template <typename Params>
        struct RandomPoint {
            template <typename F>
            Eigen::VectorXd operator()(const F& f)
            {
                return (Eigen::VectorXd::Random(f.param_size()).array() + 1) / 2;
            }
        };
    }
}

#endif