#ifndef LIMBO_BAYES_OPT_BOPTIMIZER_HPP
#define LIMBO_BAYES_OPT_BOPTIMIZER_HPP

#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/parameter/aux_/void.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <Eigen/Core>

#include <limbo/bayes_opt/bo_base.hpp>

namespace limbo {
    namespace bayes_opt {

        template <class Params, class A1 = boost::parameter::void_,
            class A2 = boost::parameter::void_, class A3 = boost::parameter::void_,
            class A4 = boost::parameter::void_, class A5 = boost::parameter::void_,
            class A6 = boost::parameter::void_>
        class BOptimizer : public BoBase<Params, A1, A2, A3, A4, A5, A6> {
        public:
            typedef BoBase<Params, A1, A2, A3, A4, A5, A6> base_t;
            typedef typename base_t::model_t model_t;
            typedef typename base_t::acquisition_function_t acquisition_function_t;
            typedef typename base_t::acqui_optimizer_t acqui_optimizer_t;

            template <typename AcquisitionFunction, typename AggregatorFunction>
            struct AcquiOptimization {
            public:
                AcquiOptimization(const AcquisitionFunction& acqui, const AggregatorFunction& afun, const Eigen::VectorXd& init) : _acqui(acqui), _afun(afun), _init(init) {}

                double utility(const Eigen::VectorXd& params) const
                {
                    return _acqui(params, _afun);
                }

                size_t param_size() const
                {
                    return _acqui.dim_in();
                }

                const Eigen::VectorXd& init() const
                {
                    return _init;
                }

            protected:
                const AcquisitionFunction& _acqui;
                const AggregatorFunction& _afun;
                const Eigen::VectorXd _init;
            };

            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true)
            {
                this->_init(sfun, afun, reset);

                _model = model_t(StateFunction::dim_in, StateFunction::dim_out);
                if (this->_observations.size())
                    _model.compute(this->_samples, this->_observations,
                        Params::boptimizer::noise());

                acqui_optimizer_t acqui_optimizer;

                while (this->_samples.size() == 0 || !this->_stop(*this, afun)) {
                    acquisition_function_t acqui(_model, this->_current_iteration);

                    Eigen::VectorXd starting_point = (Eigen::VectorXd::Random(StateFunction::dim_in).array() + 1) / 2;
                    auto acqui_optimization = AcquiOptimization<acquisition_function_t, AggregatorFunction>(acqui, afun, starting_point);
                    Eigen::VectorXd new_sample = acqui_optimizer(acqui_optimization);
                    bool blacklisted = false;
                    try {
                        this->add_new_sample(new_sample, sfun(new_sample));
                    }
                    catch (...) {
                        this->add_new_bl_sample(new_sample);
                        blacklisted = true;
                    }

                    _model.compute(this->_samples, this->_observations,
                        Params::boptimizer::noise(), this->_bl_samples);
                    this->_update_stats(*this, afun, blacklisted);

                    std::cout << this->_current_iteration << " new point: "
                              << (blacklisted ? this->_bl_samples.back()
                                              : this->_samples.back()).transpose();
                    if (blacklisted)
                        std::cout << " value: "
                                  << "No data, blacklisted";
                    else
                        std::cout << " value: " << this->_observations.back().transpose();

                    // std::cout << " mu: "<< _model.mu(blacklisted ? this->_bl_samples.back()
                    // : this->_samples.back()).transpose()
                    //<< " mean: " << _model.mean_function()(new_sample, _model).transpose()
                    //<< " sigma: "<< _model.sigma(blacklisted ? this->_bl_samples.back() :
                    //this->_samples.back())
                    //<< " acqui: "<< acqui(blacklisted ? this->_bl_samples.back() :
                    //this->_samples.back(), afun)
                    std::cout << " best:" << this->best_observation(afun) << std::endl;

                    this->_current_iteration++;
                    this->_total_iterations++;
                }
            }

            template <typename AggregatorFunction = FirstElem>
            typename AggregatorFunction::result_type best_observation(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = boost::adaptors::transform(this->_observations, afun);
                return *std::max_element(rewards.begin(), rewards.end());
            }

            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_sample(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = boost::adaptors::transform(this->_observations, afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_samples[std::distance(rewards.begin(), max_e)];
            }

            const model_t& model() const { return _model; }

        protected:
            model_t _model;
        };
    }
}

#endif
