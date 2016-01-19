#ifndef LIMBO_BAYES_OPT_PAREGO_HPP
#define LIMBO_BAYES_OPT_PAREGO_HPP

#include <algorithm>

#include <limbo/tools/macros.hpp>
#include <limbo/experimental/bayes_opt/bo_multi.hpp>

namespace limbo {
    namespace defaults {
        struct bayes_opt_parego {
            BO_PARAM(double, noise, 1e-6);
            BO_PARAM(double, rho, 0.05);
        };
    }
    namespace experimental {
        namespace bayes_opt {
            // clang-format off
            template <class Params,
                      class A3 = boost::parameter::void_,
                      class A4 = boost::parameter::void_,
                      class A5 = boost::parameter::void_,
                      class A6 = boost::parameter::void_>
            // clang-format on
            class Parego : public BoMulti<Params, A3, A4, A5, A6> {
            public:
                typedef limbo::bayes_opt::BoBase<Params, A3, A4, A5, A6> base_t;
                typedef typename base_t::model_t model_t;
                typedef typename base_t::acqui_optimizer_t acqui_optimizer_t;
                typedef typename base_t::acquisition_function_t acquisition_function_t;
                typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> pareto_point_t;
                typedef std::vector<pareto_point_t> pareto_t;

                template <typename EvalFunction>
                void optimize(const EvalFunction& feval, bool reset = true)
                {
                    this->_init(feval, FirstElem(), reset);

                    std::vector<double> scalarized = _scalarize_obs();
                    model_t model(EvalFunction::dim);
                    model.compute(this->_samples, scalarized, Params::bayes_opt_parego::noise());

                    acqui_optimizer_t inner_optimization;

                    while (this->_samples.size() == 0 || !this->_stop(*this, FirstElem())) {
                        acquisition_function_t acqui(model, this->_current_iteration);

                        Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim(), FirstElem());
                        this->add_new_sample(new_sample, feval(new_sample));
                        std::cout << this->_current_iteration
                                  << " | new sample:" << new_sample.transpose() << " => "
                                  << feval(new_sample).transpose() << std::endl;
                        scalarized = _scalarize_obs();
                        model.compute(this->_samples, scalarized, Params::bayes_opt_parego::noise());
                        this->_update_stats(*this, FirstElem(), false);
                        this->_current_iteration++;
                        this->_total_iterations++;
                    }
                    this->template update_pareto_model<EvalFunction::dim>();
                    this->update_pareto_data();
                }

            protected:
                std::vector<double> _scalarize_obs()
                {
                    assert(this->_observations.size() != 0);

                    Eigen::VectorXd lambda = Eigen::VectorXd::Random(this->_observations[0].size());
                    lambda = (lambda.array() + 1.0) / 2.0;
                    double sum = lambda.sum();
                    lambda = lambda / sum;
                    // scalarize (Tchebycheff)
                    std::vector<double> scalarized;
                    for (auto x : this->_observations) {
                        double y = (lambda.array() * x.array()).maxCoeff();
                        double s = (lambda.array() * x.array()).sum();
                        scalarized.push_back(y + Params::bayes_opt_parego::rho() * s);
                    }
                    return scalarized;
                }
            };
        }
    }
}

#endif
