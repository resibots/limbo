#ifndef BOPTIMIZER_HPP_
#define BOPTIMIZER_HPP_

#include "bo_base.hpp"
#include <boost/range/adaptor/transformed.hpp>

namespace limbo {
    struct NoReward {
        typedef double result_type;
        double operator()(const Eigen::VectorXd& x) const
        {            
            return x(0);
        }
    };

    template <class Params, class A1 = boost::parameter::void_,
        class A2 = boost::parameter::void_, class A3 = boost::parameter::void_,
        class A4 = boost::parameter::void_, class A5 = boost::parameter::void_,
        class A6 = boost::parameter::void_, class A7 = boost::parameter::void_>
    class BOptimizer : public BoBase<Params, A1, A2, A3, A4, A5, A6, A7> {
    public:
        typedef BoBase<Params, A1, A2, A3, A4, A5, A6, A7> base_t;
        typedef typename base_t::obs_t obs_t;
        typedef typename base_t::model_t model_t;
        typedef typename base_t::inner_optimization_t inner_optimization_t;
        typedef typename base_t::acquisition_function_t acquisition_function_t;

        template <typename StateFunction, typename RewardFunction = NoReward>
        void optimize(const StateFunction& sfun, const RewardFunction& rfun = RewardFunction(), bool reset = true)
        {
            this->_init(sfun, reset);

            _model = model_t(StateFunction::dim_in, StateFunction::dim_out);
            if (this->_observations.size())
                _model.compute(this->_samples, this->_observations,
                    Params::boptimizer::noise());
            inner_optimization_t inner_optimization;

            while (this->_samples.size() == 0 || this->_pursue(*this)) {
                acquisition_function_t acqui(_model, this->_iteration);

                Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim_in(), rfun);
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
                //this->_samples.back())
                std::cout << " best:" << this->best_observation(rfun) << std::endl;

                this->_iteration++;
            }
        }

        template <typename RewardFunction = NoReward>
        typename RewardFunction::result_type best_observation(const RewardFunction& rfun = RewardFunction()) const
        {
            auto rewards = boost::adaptors::transform(this->_observations, rfun);
            return *std::max_element(rewards.begin(), rewards.end());
        }

        template <typename RewardFunction = NoReward>
        const Eigen::VectorXd& best_sample(const RewardFunction& rfun = RewardFunction()) const
        {
            auto rewards = boost::adaptors::transform(this->_observations, rfun);
            auto max_e = std::max_element(rewards.begin(), rewards.end());
            return this->_samples[std::distance(rewards.begin(), max_e)];
        }

        const model_t& model() const { return _model; }

    protected:
        model_t _model;
    };
}

#endif