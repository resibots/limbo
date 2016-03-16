#ifndef LIMBO_INIT_RANDOM_SAMPLING_GRID_HPP
#define LIMBO_INIT_RANDOM_SAMPLING_GRID_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace defaults {
        struct init_randomsamplinggrid {
            BO_PARAM(int, samples, 10);
            BO_PARAM(int, bins, 5);
        };
    }
    namespace init {
        template <typename Params>
        struct RandomSamplingGrid {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                tools::rgen_int_t rgen(0, Params::init_randomsamplinggrid::bins());
                for (int i = 0; i < Params::init_randomsamplinggrid::samples(); i++) {
                    Eigen::VectorXd new_sample(StateFunction::dim_in);
                    for (size_t i = 0; i < StateFunction::dim_in; i++)
                        new_sample[i] = rgen.rand() / double(Params::init_randomsamplinggrid::bins());
                    opt.eval_and_add(seval, new_sample);
                }
            }
        };
    }
}

#endif
