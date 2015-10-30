#ifndef LIMBO_INNER_OPTI_RANDOM_HPP
#define LIMBO_INNER_OPTI_RANDOM_HPP

#include <Eigen/Core>

namespace limbo {
    namespace inner_opt {
        template <typename Params>
        struct Random {
            Random() {}

            template <typename AcquisitionFunction, typename AggregatorFunction>
            Eigen::VectorXd operator()(const AcquisitionFunction& acqui, int dim_in, const AggregatorFunction&) const
            {
                return (Eigen::VectorXd::Random(dim_in).array() + 1) / 2;
            }
        };
    }
}

#endif
