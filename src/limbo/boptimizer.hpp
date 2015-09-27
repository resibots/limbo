#ifndef BOPTIMIZER_HPP_
#define BOPTIMIZER_HPP_

#include "bo_base.hpp"

namespace limbo {
  bool compareVectorXd(Eigen::VectorXd i, Eigen::VectorXd j) { return i(0)<j(0); }

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
      this->_init(feval, reset);

      _model = model_t(EvalFunction::dim_in, EvalFunction::dim_out);
      if(this->_observations.size())
	 _model.compute(this->_samples, this->_observations, Params::boptimizer::noise());
      inner_optimization_t inner_optimization;

      while (this->_samples.size() == 0 || this->_pursue(*this)) {
        acquisition_function_t acqui(_model, this->_iteration);

        Eigen::VectorXd new_sample = inner_optimization(acqui, acqui.dim_in());
        this->add_new_sample(new_sample, feval(new_sample));	

        _model.compute(this->_samples, this->_observations, Params::boptimizer::noise());
        this->_update_stats(*this);

        std::cout << this->_iteration << " new point: "
                        << this->_samples[this->_samples.size() - 1].transpose()
                        << " value: " << this->_observations[this->_observations.size() - 1].transpose()
                        //<< " mu: "<< _model.mu(this->_samples[this->_samples.size() - 1]).transpose()
                        //<< " mean: " << _model.mean_function()(new_sample,_model).transpose()
                        //<< " sigma: "<< _model.sigma(this->_samples[this->_samples.size() - 1])
                        //<< " acqui: "<< acqui(this->_samples[this->_samples.size() - 1])
                        << " best:" << this->best_observation().transpose()
                        << std::endl;

        this->_iteration++;
      }
    }

    const obs_t& best_observation() const {
      return *std::max_element(this->_observations.begin(), this->_observations.end(), compareVectorXd);
    }

    const Eigen::VectorXd& best_sample() const {
      auto max_e = std::max_element(this->_observations.begin(), this->_observations.end(), compareVectorXd);
      return this->_samples[std::distance(this->_observations.begin(), max_e)];
    }

    const model_t& model() const {
      return _model;
    }

  protected: 
    model_t _model;

  };
}

#endif