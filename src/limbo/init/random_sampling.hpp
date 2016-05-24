#ifndef LIMBO_INIT_RANDOM_SAMPLING_HPP
#define LIMBO_INIT_RANDOM_SAMPLING_HPP

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace defaults {
        struct init_randomsampling {
            ///@ingroup init_defaults
            BO_PARAM(int, samples, 10);
        };
    }
    namespace init {
        /** @ingroup init
          \rst
          Pure random sampling in [0, 1]^n

          Parameters:
            - ``int samples`` (total number of samples)
          \endrst
        */
        template <typename Params>
        struct RandomSampling {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                for (int i = 0; i < Params::init_randomsampling::samples(); i++) {
                    auto new_sample = tools::random_vector(StateFunction::dim_in);
                    opt.eval_and_add(seval, new_sample);
                }
            }
        };
    }
}

#endif
