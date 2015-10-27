#ifndef INITIALIZATION_FUNCTIONS_RANDOM_SAMPLING_GRID_HPP_
#define INITIALIZATION_FUNCTIONS_RANDOM_SAMPLING_GRID_HPP_

#include <Eigen/Core>

namespace limbo {
    namespace initialization_functions {
        // initialize in [0,1] !
        // params:
        //  -init::nb_bins
        //  - init::nb_samples
        template <typename Params>
        struct RandomSamplingGrid {
            template <typename F, typename Opt>
            void operator()(const F& feval, Opt& opt) const
            {
                for (int i = 0; i < Params::init::nb_samples(); i++) {
                    Eigen::VectorXd new_sample(F::dim_in);
                    for (size_t i = 0; i < F::dim_in; i++)
                        new_sample[i] = int(((double)(Params::init::nb_bins() + 1) * rand()) / (RAND_MAX + 1.0)) / double(Params::init::nb_bins());
                    opt.add_new_sample(new_sample, feval(new_sample));
                }
            }
        };
    }
}
#endif
