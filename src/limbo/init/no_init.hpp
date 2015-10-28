#ifndef LIMBO_INIT_NO_INIT_HPP
#define LIMBO_INIT_NO_INIT_HPP

namespace limbo {
    namespace init {
        // params is here only to make it easy to switch
        // from/to the other init functions
        template <typename Params>
        struct NoInit {
            template <typename F, typename Opt>
            void operator()(const F& f, Opt& opt) const {}
        };
    }
}

#endif
