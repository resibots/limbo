#ifndef LIMBO_INIT_RANDOM_SAMPLING_HPP
#define LIMBO_INIT_RANDOM_SAMPLING_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/math.hpp>

namespace limbo {
    namespace defaults {
        struct init_randomsampling {
            BO_PARAM(int, samples, 10);
        };
    }
    namespace init {
        template <typename Params>
        struct RandomSampling {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                tools::rgen_double_t rgen(0.0, 1.0);
                for (int i = 0; i < Params::init_randomsampling::samples(); i++) {
                    Eigen::VectorXd new_sample(StateFunction::dim_in);
                    for (size_t j = 0; j < StateFunction::dim_in; j++)
                        new_sample[j] = rgen.rand();
                    opt.add_new_sample(new_sample, seval(new_sample));
                }
            }
        };
    }
}

#endif
