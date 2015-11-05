#ifndef LIMBO_STOP_MAX_PREDICTED_VALUE_HPP
#define LIMBO_STOP_MAX_PREDICTED_VALUE_HPP

#include <iostream>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>

namespace limbo {

    namespace defaults {
        struct maxpredictedvalue {
            BO_PARAM(float, ratio, 0.9);
        };
    }

    namespace stop {
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
                typename BO::acqui_optimizer_t opti;
                typename BO::acquisition_function_t acqui(bo.model(), bo.iteration());
                Eigen::VectorXd starting_point = (Eigen::VectorXd::Random(acqui.dim_in()).array() + 1) / 2;
                auto acqui_optimization = typename BO::template AcquiOptimization<typename BO::acquisition_function_t, AggregatorFunction>(acqui, afun, starting_point);
                double val = gpmean(opti(acqui_optimization), afun);

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
    }
}

#endif
