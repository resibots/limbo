#ifndef INITIALIZATION_FUNCTIONS_NO_INIT_HPP_
#define INITIALIZATION_FUNCTIONS_NO_INIT_HPP_

namespace limbo {
    namespace initialization_functions {
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
