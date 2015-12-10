#ifndef LIMBO_OPT_PARALLEL_REPEATER_HPP
#define LIMBO_OPT_PARALLEL_REPEATER_HPP

#include <algorithm>

#include <boost/math/special_functions/sign.hpp>

#include <Eigen/Core>

#include <limbo/tools/parallel.hpp>

namespace limbo {
    namespace defaults {
        struct parallel_repeater {
            BO_PARAM(int, repeats, 10);
        };
    }
    namespace opt {
        template <typename Params, typename Optimizer>
        struct ParallelRepeater {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, bool bounded) const
            {
                tools::par::init();
                typedef std::pair<Eigen::VectorXd, double> pair_t;
                auto body = [&](int i) {
                    // clang-format off
                    Eigen::VectorXd v = Optimizer()(f, bounded);

                    double lik = f.utility(v);
                    return std::make_pair(v, lik);
                    // clang-format on
                };

                auto comp = [](const pair_t& v1, const pair_t& v2) {
                    // clang-format off
                    return v1.second > v2.second;
                    // clang-format on
                };

                pair_t init = std::make_pair(f.init(), -std::numeric_limits<float>::max());
                auto m = tools::par::max(init, Params::parallel_repeater::repeats(), body, comp);

                return m.first;
            };
        };
    }
}

#endif
