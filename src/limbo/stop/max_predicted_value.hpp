#ifndef LIMBO_STOP_MAX_PREDICTED_VALUE_HPP
#define LIMBO_STOP_MAX_PREDICTED_VALUE_HPP

#include <iostream>

#include <boost/parameter/aux_/void.hpp>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace defaults {
        struct stop_maxpredictedvalue {
            BO_PARAM(double, ratio, 0.9);
        };
    }
    namespace stop {
        template <typename Params, typename Optimizer = boost::parameter::void_>
        struct MaxPredictedValue {

            MaxPredictedValue() {}

            template <typename BO, typename AggregatorFunction>
            bool operator()(const BO& bo, const AggregatorFunction& afun) const
            {
                // Prevent instantiation of GPMean if there are no observed samplesgit
                if (bo.observations().size() == 0)
                    return false;

                auto optimizer = _get_optimizer(typename BO::acqui_optimizer_t(), Optimizer());
                Eigen::VectorXd starting_point = tools::random_vector(bo.model().dim_in());
                auto model_optimization =
                    [&](const Eigen::VectorXd& x, bool g) { return opt::no_grad(afun(bo.model().mu(x))); };
                auto x = optimizer(model_optimization, starting_point, true);
                double val = afun(bo.model().mu(x));

                if (bo.observations().size() == 0 || afun(bo.best_observation(afun)) <= Params::stop_maxpredictedvalue::ratio() * val)
                    return false;
                else {
                    std::cout << "stop caused by Max predicted value reached. Threshold: "
                              << Params::stop_maxpredictedvalue::ratio() * val
                              << " max observations: " << afun(bo.best_observation(afun)) << std::endl;
                    return true;
                }
            }

        protected:
            template <typename Model, typename AggregatorFunction>
            struct ModelMeanOptimization {
            public:
                ModelMeanOptimization(const Model& model, const AggregatorFunction& afun, const Eigen::VectorXd& init) : _model(model), _afun(afun), _init(init) {}

                double utility(const Eigen::VectorXd& v) const
                {
                    return _afun(_model.mu(v));
                }

                size_t param_size() const
                {
                    return _model.dim_in();
                }

                const Eigen::VectorXd& init() const
                {
                    return _init;
                }

            protected:
                const Model& _model;
                const AggregatorFunction& _afun;
                const Eigen::VectorXd _init;
            };

            template <typename Model, typename AggregatorFunction>
            inline ModelMeanOptimization<Model, AggregatorFunction> _make_model_mean_optimization(const Model& model, const AggregatorFunction& afun, const Eigen::VectorXd& init) const
            {
                return ModelMeanOptimization<Model, AggregatorFunction>(model, afun, init);
            }

            template <typename BoAcquiOpt>
            inline BoAcquiOpt _get_optimizer(const BoAcquiOpt& bo_acqui_opt, boost::parameter::void_) const
            {
                return bo_acqui_opt;
            }

            template <typename BoAcquiOpt, typename CurrentOpt>
            inline CurrentOpt _get_optimizer(const BoAcquiOpt&, const CurrentOpt& current_opt) const
            {
                return current_opt;
            }
        };
    }
}

#endif
