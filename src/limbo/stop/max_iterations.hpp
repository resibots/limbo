#ifndef LIMBO_STOP_MAX_ITERATIONS_HPP
#define LIMBO_STOP_MAX_ITERATIONS_HPP

#include <limbo/tools/macros.hpp>

namespace limbo {
    namespace defaults {
        struct stop_maxiterations {
            /// @ingroup stop_defaults 
            BO_PARAM(int, iterations, 190);
        };
    }
    namespace stop {
        /// @ingroup stop
        /// Stop after a given number of iterations
        ///
        /// parameter: int iterations
        template <typename Params>
        struct MaxIterations {
            MaxIterations() {}

            template <typename BO, typename AggregatorFunction>
            bool operator()(const BO& bo, const AggregatorFunction&)
            {
                return bo.current_iteration() >= Params::stop_maxiterations::iterations();
            }
        };
    }
}

#endif
