#ifndef LIMBO_BAYES_OPT_BOPTIMIZER_HPP
#define LIMBO_BAYES_OPT_BOPTIMIZER_HPP

#include <iostream>
#include <algorithm>
#include <iterator>

#include <boost/parameter/aux_/void.hpp>

#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/tools/random_generator.hpp>
#include <limbo/bayes_opt/bo_base.hpp>

namespace limbo {
    namespace defaults {
        struct bayes_opt_boptimizer {
            BO_PARAM(double, noise, 1e-6);
        };
    }

    BOOST_PARAMETER_TEMPLATE_KEYWORD(acquiopt)

    namespace bayes_opt {

        typedef boost::parameter::parameters<boost::parameter::optional<tag::acquiopt>,
            boost::parameter::optional<tag::statsfun>,
            boost::parameter::optional<tag::initfun>,
            boost::parameter::optional<tag::acquifun>,
            boost::parameter::optional<tag::stopcrit>,
            boost::parameter::optional<tag::modelfun>> boptimizer_signature;

        // clang-format off
        template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_,
          class A6 = boost::parameter::void_>
        // clang-format on
        class BOptimizer : public BoBase<Params, A1, A2, A3, A4, A5> {
        public:
            // defaults
            struct defaults {
#ifdef USE_LIBCMAES
                typedef opt::Cmaes<Params> acquiopt_t; // 2
#elif defined(USE_NLOPT)
                typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND> acquiopt_t;
#else
#warning NO NLOpt, and NO Libcmaes: the acquisition function will be optimized by a grid search algorithm (which is usually bad). Please install at least NLOpt or libcmaes to use limbo!.
                typedef opt::GridSearch<Params> acquiopt_t;
#endif
            };

            typedef BoBase<Params, A1, A2, A3, A4, A5> base_t;
            typedef typename base_t::model_t model_t;
            typedef typename base_t::acquisition_function_t acquisition_function_t;
            // extract the types
            typedef typename boptimizer_signature::bind<A1, A2, A3, A4, A5, A6>::type args;
            typedef typename boost::parameter::binding<args, tag::acquiopt, typename defaults::acquiopt_t>::type acqui_optimizer_t;

            template <typename StateFunction, typename AggregatorFunction = FirstElem>
            void optimize(const StateFunction& sfun, const AggregatorFunction& afun = AggregatorFunction(), bool reset = true)
            {

                this->_init(sfun, afun, reset);

                if (!this->_observations.empty())
                    _model.compute(this->_samples, this->_observations, Params::bayes_opt_boptimizer::noise(), this->_bl_samples);
                else
                    _model = model_t(StateFunction::dim_in, StateFunction::dim_out);

                acqui_optimizer_t acqui_optimizer;

                while (!this->_stop(*this, afun)) {
                    acquisition_function_t acqui(_model, this->_current_iteration);

                    // we do not have gradient in our current acquisition function
                    auto acqui_optimization =
                        [&](const Eigen::VectorXd& x, bool g) { return opt::no_grad(acqui(x, afun)); };
                    Eigen::VectorXd starting_point = tools::random_vector(StateFunction::dim_in);
                    Eigen::VectorXd new_sample = acqui_optimizer(acqui_optimization, starting_point, true);
                    bool blacklisted = !this->eval_and_add(sfun, new_sample);

                    this->_update_stats(*this, afun, blacklisted);

                    if (!this->_observations.empty())
                        _model.compute(this->_samples, this->_observations, Params::bayes_opt_boptimizer::noise(), this->_bl_samples);

                    this->_current_iteration++;
                    this->_total_iterations++;
                }
            }

            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_observation(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_observations.size());
                std::transform(this->_observations.begin(), this->_observations.end(), rewards.begin(), afun);
                auto max_e = std::max_element(rewards.begin(), rewards.end());
                return this->_observations[std::distance(rewards.begin(), max_e)];
            }

            template <typename AggregatorFunction = FirstElem>
            const Eigen::VectorXd& best_sample(const AggregatorFunction& afun = AggregatorFunction()) const
            {
                auto rewards = std::vector<double>(this->_observations.size());
                std::transform(this->_observations.begin(), this->_observations.end(), rewards.begin(), afun);
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
