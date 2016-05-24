#ifndef LIMBO_INIT_NO_INIT_HPP
#define LIMBO_INIT_NO_INIT_HPP

namespace limbo {
    namespace init {
        ///@ingroup init
        ///Do nothing (dummy initializer).
        template <typename Params>
        struct NoInit {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction&, const AggregatorFunction&, Opt&) const {}
        };
    }
}

#endif
