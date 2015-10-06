#ifndef PAREGO_HPP_
#define PAREGO_HPP_

#include <algorithm>
#include "limbo.hpp"
#include "bo_multi.hpp"

namespace limbo {
  namespace defaults {
    struct parego {
      BO_PARAM(double, rho, 0.05);
    };
  }

  template <
    class Params
    , class A3 = boost::parameter::void_
    , class A4 = boost::parameter::void_
    , class A5 = boost::parameter::void_
    , class A6 = boost::parameter::void_
    , class A7 = boost::parameter::void_
    >
  class Parego : public BoMulti<Params, A3, A4, A5, A6, A7> {
   public:
    typedef BoBase<Params, obs_type<Eigen::VectorXd>, A3, A4, A5, A6, A7> base_t;
    typedef typename base_t::obs_t obs_t;
    typedef typename base_t::model_t model_t;
    typedef typename base_t::inner_optimization_t inner_optimization_t;
    typedef typename base_t::acquisition_function_t acquisition_function_t;
    typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, double> pareto_point_t;
    typedef std::vector<pareto_point_t> pareto_t;

    template<typename EvalFunction>
    void optimize(const EvalFunction& feval, bool reset = true) {
      this->_init(feval, reset);

      std::vector<double> scalarized = _scalarize_obs();
      model_t model(EvalFunction::dim);
      model.compute(this->_samples, scalarized, Params::boptimizer::noise());

      inner_optimization_t inner_optimization;

      while (this->_samples.size() == 0 || this->_pursue(*this)) {
        acquisition_function_t acqui(model, this->_iteration);

        Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim());
        this->add_new_sample(new_sample, feval(new_sample));
        std::cout << this->_iteration
                  << " | new sample:" << new_sample.transpose()
                  << " => " << feval(new_sample).transpose() << std::endl;
        scalarized = _scalarize_obs();
        model.compute(this->_samples, scalarized, Params::boptimizer::noise());
        this->_update_stats(*this);
        this->_iteration++;
      }
      this->template update_pareto_model<EvalFunction::dim>();
      this->update_pareto_data();
    }

   protected:

    std::vector<double> _scalarize_obs() {
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
        scalarized.push_back(y + Params::parego::rho() * s);
      }
      return scalarized;
    }

  };







}

#endif
