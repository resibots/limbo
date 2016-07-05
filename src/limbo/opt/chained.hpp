#ifndef LIMBO_OPT_CHAINED_HPP
#define LIMBO_OPT_CHAINED_HPP

#include <algorithm>

#include <Eigen/Core>

#include <limbo/opt/optimizer.hpp>

namespace limbo {
    namespace opt {

        // Needed for the variadic data structure
        template <typename Params, typename... Optimizers>
        struct Chained {
        };

        // Base case: just 1 optimizer to call
        template <typename Params, typename Optimizer>
        struct Chained<Params, Optimizer> {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                return Optimizer()(f, init, bounded);
            };
        };

        // Recursive case: call current optimizer, and pass result as init value for the next one
        template <typename Params, typename Optimizer, typename... Optimizers>
        struct Chained<Params, Optimizer, Optimizers...> : Chained<Params, Optimizers...> {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                return Chained<Params, Optimizers...>::operator()(f, Optimizer()(f, init, bounded), bounded);
            };
        };
    }
}

#endif
