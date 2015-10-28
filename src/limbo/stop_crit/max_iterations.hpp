#ifndef STOPPING_CRITERIA_MAX_ITERATIONS_HPP_
#define STOPPING_CRITERIA_MAX_ITERATIONS_HPP_

namespace limbo {
    namespace stop_crit {
        template <typename Params>
        struct MaxIterations {
            MaxIterations() {}

            template <typename BO, typename AggregatorFunction>
            bool operator()(const BO& bo, const AggregatorFunction&)
            {
                return bo.iteration() <= Params::maxiterations::n_iterations();
            }
        };
    }
}
#endif
