#ifndef LIMBO_INIT_RANDOM_SAMPLING_GRID_HPP
#define LIMBO_INIT_RANDOM_SAMPLING_GRID_HPP

#include <Eigen/Core>

namespace limbo {
    namespace init {
        // initialize in [0,1] !
        // params:
        //  -init::nb_bins
        //  - init::nb_samples
        template <typename Params>
        struct RandomSamplingGrid {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                for (int i = 0; i < Params::init::nb_samples(); i++) {
                    Eigen::VectorXd new_sample(StateFunction::dim_in);
                    for (size_t i = 0; i < StateFunction::dim_in; i++)
                        new_sample[i] = int(((double)(Params::init::nb_bins() + 1) * rand()) / (RAND_MAX + 1.0)) / double(Params::init::nb_bins());
                    opt.add_new_sample(new_sample, seval(new_sample));
                }
            }
        };
    }
}

#endif
