#ifndef EHVI_HPP_
#define EHVI_HPP_

#include <algorithm>
#include "bo_multi.hpp"
#include "ehvi/ehvi_calculations.h"
#include "ehvi/ehvi_sliceupdate.h"

namespace limbo {
  namespace acquisition_functions {
    // only work in 2D for now
    template<typename Params, typename Model>
    class Ehvi {
     public:
      Ehvi(const std::vector<Model>& models,
           const std::deque<individual*>& pop,
           const Eigen::VectorXd& ref_point)
        : _models(models), _pop(pop), _ref_point(ref_point) {
        assert(_models.size() == 2);
      }
      size_t dim() const {
        return _models[0].dim();
      }
      double operator()(const Eigen::VectorXd& v) const {
        assert(_models.size() == 2);
        double r[3] = { _ref_point(0), _ref_point(1), _ref_point(2) };
        double mu[3] = { _models[0].mu(v), _models[1].mu(v), 0 };
        double s[3] = { _models[0].sigma(v), _models[1].sigma(v), 0 };
        //for (size_t i = 0; i < _models.size(); ++i)
//          mu[i] = std::min(_models[i].mu(v), _models[i].max_observation());
          double ehvi = ehvi2d(_pop, r, mu, s);
        return ehvi;
      }
     protected:
      const std::vector<Model>& _models;
      const std::deque<individual*>& _pop;
      Eigen::VectorXd _ref_point;
    };
  }

  template <
    class Params
    , class A2 = boost::parameter::void_
    , class A3 = boost::parameter::void_
    , class A4 = boost::parameter::void_
    , class A5 = boost::parameter::void_
    , class A6 = boost::parameter::void_
    , class A7 = boost::parameter::void_
    >
  class Ehvi : public BoMulti<Params, A2, A3, A4, A5, A6, A7> {
   public:
    typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> pareto_point_t;
    typedef BoBase<Params, obs_type<Eigen::VectorXd>, A3, A4, A5, A6, A7> base_t;
    typedef typename base_t::model_t model_t;
    typedef typename base_t::inner_optimization_t inner_optimization_t;

    template<typename EvalFunction>
    void optimize(const EvalFunction& feval, bool reset = true) {
      this->_init(feval, reset);

      inner_optimization_t inner_opt;

      while (this->_samples.size() == 0 || this->_pursue()) {
        std::cout << "updating models...";
        std::cout.flush();
        this->_update_models();
        this->update_pareto_data();
        std::cout << "ok" << std::endl;

        // copy in the ehvi structure to compute expected improvement
        std::deque<individual*> pop;
        for (auto x : this->pareto_data()) {
          individual* ind = new individual;
          ind->f[0] = std::get<1>(x)(0);
          ind->f[1] = std::get<1>(x)(1);
          ind->f[2] = 0;
          pop.push_back(ind);
        }

        // optimize ehvi
        std::cout << "optimizing ehvi" << std::endl;

          auto acqui =
            acquisition_functions::Ehvi<Params, model_t>
              (this->_models, pop,
              Eigen::Vector3d(Params::ehvi::x_ref(), Params::ehvi::y_ref(), 0));

        double best_hv = -1;
        Eigen::VectorXd best_s;
        for (auto x : this->pareto_data()) {
          Eigen::VectorXd s = inner_opt(acqui, acqui.dim(), std::get<0>(x));
          double hv = acqui(s);
          if (hv > best_hv)
            {
              best_s = s;
              best_hv = hv;
            }
        }
        std::cout<<"sample selected" << std::endl;
        Eigen::VectorXd new_sample = best_s;
        std::cout<<"new sample:"<<new_sample.transpose()<<std::endl;

        std::cout<<"expected improvement: "<<acqui(new_sample)<<std::endl;
        std::cout<<"expected value: "<<this->_models[0].mu(new_sample)
                <<" "<<this->_models[1].mu(new_sample)
                <<" "<<this->_models[0].sigma(new_sample)
                <<" "<<this->_models[1].sigma(new_sample)
                 << std::endl;
        std::cout << "opt done" << std::endl;


        // delete pop
        for (auto x : pop)
          delete x;

        // add sample
        this->add_new_sample(new_sample, feval(new_sample));
        std::cout << this->_iteration
                  << " | new sample:" << new_sample.transpose()
                  << " => " << feval(new_sample).transpose() << std::endl;

        _update_stats();
        this->_iteration++;
      }

    }

   protected:
    void _update_stats() {
      boost::fusion::for_each(this->_stat, RefreshStat_f<Ehvi>(*this));
    }
  };

}

#endif
