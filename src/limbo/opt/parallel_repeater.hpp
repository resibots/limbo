#ifndef LIMBO_OPT_PARALLEL_REPEATER_HPP
#define LIMBO_OPT_PARALLEL_REPEATER_HPP

#include <algorithm>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/parallel.hpp>
#include <limbo/tools/random_generator.hpp>
#include <limbo/opt/optimizer.hpp>

namespace limbo {
    namespace defaults {
        struct opt_parallelrepeater {
            BO_PARAM(int, repeats, 10);
        };
    }
    namespace opt {
        template <typename Params, typename Optimizer>
        struct ParallelRepeater {
            template <typename F>
            Eigen::VectorXd operator()(const F& f, const Eigen::VectorXd& init, bool bounded) const
            {
                tools::par::init();
                typedef std::pair<Eigen::VectorXd, double> pair_t;
                auto body = [&](int i) {
                    // clang-format off
                    Eigen::VectorXd r_init = tools::random_vector(init.size());
                    Eigen::VectorXd v = Optimizer()(f, init, bounded);
                    double lik = opt::eval(f, v);
                    return std::make_pair(v, lik);
                    // clang-format on
                };

                auto comp = [](const pair_t& v1, const pair_t& v2) {
                    // clang-format off
                    return v1.second > v2.second;
                    // clang-format on
                };

                pair_t init_v = std::make_pair(init, -std::numeric_limits<float>::max());
                auto m = tools::par::max(init_v, Params::opt_parallelrepeater::repeats(), body, comp);

                return m.first;
            };
        };
    }
}

#endif
