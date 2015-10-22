#ifndef STOPPING_CRITERION_HPP_
#define STOPPING_CRITERION_HPP_

#include <iostream>
#include <Eigen/Core>
#include <vector>

// USING_PART_OF_NAMESPACE_EIGEN

namespace limbo {

    namespace defaults {
        struct maxpredictedvalue {
            BO_PARAM(float, ratio, 0.9);
        };
    }

    namespace stopping_criterion {
        template <typename Params>
        struct MaxIterations {
            MaxIterations() { iteration = 0; }

            template <typename BO, typename AggregatorFunction>
            bool operator()(const BO& bo, const AggregatorFunction&)
            {
                return bo.iteration() <= Params::maxiterations::n_iterations();
            }

        protected:
            int iteration;
        };

        template <typename Params>
        struct MaxPredictedValue {

            MaxPredictedValue() {}

            template <typename BO, typename AggregatorFunction>
            bool operator()(const BO& bo, const AggregatorFunction& afun)
            {
                // Prevent instantiation of GPMean if there are no observed samplesgit
                if (bo.observations().size() == 0)
                    return true;

                GPMean<BO> gpmean(bo);
                typename BO::inner_optimization_t opti;
                double val = gpmean(opti(gpmean, gpmean.dim_in(), afun), afun);

                if (bo.observations().size() == 0 || bo.best_observation(afun) <= Params::maxpredictedvalue::ratio() * val)
                    return true;
                else {
                    std::cout << "stop caused by Max predicted value reached. Thresold: "
                              << Params::maxpredictedvalue::ratio() * val
                              << " max observations: " << bo.best_observation(afun) << std::endl;
                    return false;
                }
            }

        protected:
            template <typename BO>
            struct GPMean {
                GPMean(const BO& bo)
                    : _model(bo.samples()[0].size(), bo.observations()[0].size())
                { // should have at least one sample
                    _model.compute(bo.samples(), bo.observations(),
                        BO::params_t::boptimizer::noise());
                }

                template <typename AggregatorFunction>
                double operator()(const Eigen::VectorXd& v, const AggregatorFunction afun) const { return afun(_model.mu(v)); }

                int dim_in() const
                {
                    return _model.dim_in();
                }

            protected:
                typename BO::model_t _model;
            };
        };

        template <typename BO, typename AggregatorFunction>
        struct ChainCriteria {
            typedef bool result_type;
            ChainCriteria(const BO& bo, const AggregatorFunction& afun) : _bo(bo), _afun(afun) {}

            template <typename stopping_criterion>
            bool operator()(bool state, stopping_criterion stop) const
            {
                return state && stop(_bo, _afun);
            }

        protected:
            const BO& _bo;
            const AggregatorFunction& _afun;
        };
    }
}
#endif
