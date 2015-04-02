#ifndef BOPTIMIZER_HPP_
#define BOPTIMIZER_HPP_

#include <type_traits>
#include "bo_base.hpp"


namespace limbo {

  template <
    class Params
    , class A1 = boost::parameter::void_
    , class A2 = boost::parameter::void_
    , class A3 = boost::parameter::void_
    , class A4 = boost::parameter::void_
    , class A5 = boost::parameter::void_
    , class A6 = boost::parameter::void_
    , class A7 = boost::parameter::void_
    >
  class BOptimizer : public BoBase<Params, A1, A2, A3, A4, A5, A6, A7> {
   public:
    typedef BoBase<Params, A1, A2, A3, A4, A5, A6, A7> base_t;
    typedef typename base_t::obs_t obs_t;
    typedef typename base_t::model_t model_t;
    typedef typename base_t::inner_optimization_t inner_optimization_t;
    typedef typename base_t::acquisition_function_t acquisition_function_t;

    template<typename EvalFunction>
    void optimize(const EvalFunction& feval, bool reset = true) {
      static_assert(std::is_floating_point<obs_t>::value, "BOptimizer wants double/double for obs");
      this->_init(feval, reset);
      model_t model(EvalFunction::dim);

      inner_optimization_t inner_optimization;

      while (this->_samples.size() == 0 || this->_pursue(*this)) {
        acquisition_function_t acqui(model, this->_iteration);

        Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim());
        this->add_new_sample(new_sample, feval(new_sample));

        model.compute(this->_samples, this->_observations, Params::boptimizer::noise());
        this->_update_stats(*this);

        std::cout << this->_iteration << " new point: "
                  << this->_samples[this->_samples.size() - 1].transpose()
                  << " value: " << this->_observations[this->_observations.size() - 1]
                  << " best:" << this->best_observation()
                  << std::endl;

        this->_iteration++;
      }
    }

    const obs_t& best_observation() const {
      return *std::max_element(this->_observations.begin(), this->_observations.end());
    }

    const Eigen::VectorXd& best_sample() const {
      auto max_e = std::max_element(this->_observations.begin(), this->_observations.end());
      return this->_samples[std::distance(this->_observations.begin(), max_e)];
    }

  };








}

#endif
