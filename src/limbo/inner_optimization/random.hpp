#ifndef INNER_OPTIMIZATION_RANDOM_HPP_
#define INNER_OPTIMIZATION_RANDOM_HPP_

#include <Eigen/Core>

namespace limbo {
    namespace inner_optimization {
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
