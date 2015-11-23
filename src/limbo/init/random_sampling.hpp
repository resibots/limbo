#ifndef LIMBO_INIT_RANDOM_SAMPLING_HPP
#define LIMBO_INIT_RANDOM_SAMPLING_HPP

#include <iostream>

#include <Eigen/Core>

#include <limbo/tools/rand.hpp>

namespace limbo {
    namespace init {
        // initialize in [0,1] !
        // params: init::nb_samples
        template <typename Params>
        struct RandomSampling {
            template <typename StateFunction, typename AggregatorFunction, typename Opt>
            void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
            {
                for (int i = 0; i < Params::init::nb_samples(); i++) {
                    Eigen::VectorXd new_sample(StateFunction::dim_in);
                    for (int i = 0; i < StateFunction::dim_in; i++)
                        new_sample[i] = tools::rand<double>(0, 1);
                    std::cout << "random sample:" << new_sample.transpose() << std::endl;
                    opt.add_new_sample(new_sample, seval(new_sample));
                }
            }
        };
    }
}

#endif
