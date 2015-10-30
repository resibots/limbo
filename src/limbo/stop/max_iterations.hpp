#ifndef LIMBO_STOP_MAX_ITERATIONS_HPP
#define LIMBO_STOP_MAX_ITERATIONS_HPP

namespace limbo {
    namespace stop {
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
