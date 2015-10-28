#ifndef STOPPING_CRITERIA_CHAIN_CRITERIA_HPP_
#define STOPPING_CRITERIA_CHAIN_CRITERIA_HPP_

namespace limbo {
    namespace stop {
        template <typename BO, typename AggregatorFunction>
        struct ChainCriteria {
            typedef bool result_type;
            ChainCriteria(const BO& bo, const AggregatorFunction& afun) : _bo(bo), _afun(afun) {}

            template <typename stopping_criterion>
            bool operator()(bool state, stopping_criterion stop) const
            {
                return state && stop(_bo, _afun);
            }

        protected:
            const BO& _bo;
            const AggregatorFunction& _afun;
        };
    }
}
#endif
