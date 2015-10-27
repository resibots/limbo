#ifndef INITIALIZATION_FUNCTIONS_HPP_
#define INITIALIZATION_FUNCTIONS_HPP_

#include <limbo/initialization_functions/random_sampling.hpp>
#include <limbo/initialization_functions/random_sampling_grid.hpp>
#include <limbo/initialization_functions/grid_sampling.hpp>

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
