#ifndef LIMBO_STOP_CHAIN_CRITERIA_HPP
#define LIMBO_STOP_CHAIN_CRITERIA_HPP

namespace limbo {
    namespace stop {
        /**
          \rst
          Utility functor for boost::fusion::accumulate, e.g.:

          .. code-block:: cpp

            stop::ChainCriteria<BO, AggregatorFunction> chain(bo, afun);
            return boost::fusion::accumulate(_stopping_criteria, false, chain);

          Where ``_stopping_criteria` is a ``boost::fusion::vector`` of classes.

          \endrst
        */
        template <typename BO, typename AggregatorFunction>
        struct ChainCriteria {
            typedef bool result_type;
            ChainCriteria(const BO& bo, const AggregatorFunction& afun) : _bo(bo), _afun(afun) {}

            template <typename stopping_criterion>
            bool operator()(bool state, stopping_criterion stop) const
            {
                return state || stop(_bo, _afun);
            }

        protected:
            const BO& _bo;
            const AggregatorFunction& _afun;
        };
    }
}

#endif
