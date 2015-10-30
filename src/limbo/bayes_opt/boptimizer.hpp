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
            typedef typename base_t::inner_optimization_t inner_optimization_t;
            typedef typename base_t::acquisition_function_t acquisition_function_t;

            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true)
            {
                this->_init(sfun, reset);

                _model = model_t(StateFunction::dim_in, StateFunction::dim_out);
                if (this->_observations.size())
                    _model.compute(this->_samples, this->_observations,
                        Params::boptimizer::noise());
                inner_optimization_t inner_optimization;

                while (this->_samples.size() == 0 || this->_pursue(*this, afun)) {
                    acquisition_function_t acqui(_model, this->_iteration);

                    Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim_in(), afun);
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
                    this->_update_stats(*this, blacklisted);

                    std::cout << this->_iteration << " new point: "
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

                    this->_iteration++;
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
