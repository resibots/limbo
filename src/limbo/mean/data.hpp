#ifndef LIMBO_MEAN_DATA_HPP
#define LIMBO_MEAN_DATA_HPP

#include <Eigen/Core>

namespace limbo {
    namespace mean {
        template <typename Params>
        struct Data {
            Data(size_t dim_out = 1) {}

            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP& gp) const
            {
                return gp.mean_observation().array();
            }
        };
    }
}

#endif
