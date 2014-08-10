#ifndef BO_MULTI_HPP_
#define BO_MULTI_HPP_

#include "bo_base.hpp"

namespace limbo {
  namespace multi {
    struct SferesParams {
      struct evo_float {
        typedef sferes::gen::evo_float::mutation_t mutation_t;
        typedef sferes::gen::evo_float::cross_over_t cross_over_t;
        SFERES_CONST float cross_rate = 0.5f;
        SFERES_CONST float mutation_rate = 0.1f;
        SFERES_CONST float eta_m = 15.0f;
        SFERES_CONST float eta_c = 10.0f;
        SFERES_CONST mutation_t mutation_type =
          sferes::gen::evo_float::polynomial;
        SFERES_CONST cross_over_t cross_over_type =
          sferes::gen::evo_float::sbx;
      };
      struct pop {
        SFERES_CONST unsigned size = 100;
        SFERES_CONST unsigned nb_gen = 2000;
        SFERES_CONST int dump_period = -1;
        SFERES_CONST int initial_aleat = 1;
      };
      struct parameters {
        SFERES_CONST float min = 0.0f;
        SFERES_CONST float max = 1.0f;
      };
    };

    template<typename M>
    class SferesFit {
     public:
      SferesFit(const std::vector<M>& models) : _models(models) {
      }
      SferesFit() {}
      const std::vector<float>& objs() const {
        return _objs;
      }
      float obj(size_t i) const {
        return _objs[i];
      }
      template<typename Indiv>
      void eval(const Indiv& indiv) {
        this->_objs.resize(_models.size());
        Eigen::VectorXd v(indiv.size());
        for (size_t j = 0; j < indiv.size(); ++j)
          v[j] = indiv.data(j);
        for (size_t i = 0; i < _models.size(); ++i)
          this->_objs[i] = _models[i].mu(v);
      }
     protected:
      std::vector<M> _models;
      std::vector<float> _objs;
    };
  }


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
  class BoMulti : public BoBase<Params, obs_type<Eigen::VectorXd>,
    A2, A3, A4, A5, A6, A7> {
   public:
    typedef BoBase<Params, obs_type<Eigen::VectorXd>, A2, A3, A4, A5, A6, A7> base_t;
    typedef typename base_t::obs_t obs_t;
    typedef typename base_t::model_t model_t;
    typedef typename base_t::inner_optimization_t inner_optimization_t;
    typedef typename base_t::acquisition_function_t acquisition_function_t;
    typedef std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::VectorXd> pareto_point_t;
    typedef std::vector<pareto_point_t> pareto_t;

    size_t nb_objs() const {
      return this->_observations[0].size();
    }
   protected:
    void _compute_pareto_model() {
      typedef sferes::gen::EvoFloat<EvalFunction::dim, multi::SferesParams> gen_t;
      typedef sferes::phen::Parameters<gen_t, mulit::SferesFit<model_t>, multi::SferesParams> phen_t;
      typedef sferes::eval::Parallel<multi::SferesParams> eval_t;
      typedef boost::fusion::vector<> stat_t;
      typedef sferes::modif::Dummy<> modifier_t;
      typedef sferes::ea::Nsga2<phen_t, eval_t, stat_t, modifier_t, multi::SferesParams> nsga2_t;

      nsga2_t ea;
      ea.set_fit_proto(SferesFit<model_t>(models));
      ea.run();
      auto pareto_front = ea.pareto_front();
      

    }
    void _update_models() {
      size_t dim = this->_samples[0].size();
      std::vector<std::vector<double> > uni_obs(nb_objs);
      for (size_t i = 0; i < this->_observations.size(); ++i)
        for (size_t j = 0; j < this->_observations[i].size(); ++j)
          uni_obs[j].push_back(this->_observations[i][j]);
      std::vector<model_t> models(nb_objs(), model_t(dim));
      for (size_t i = 0; i < uni_obs.size(); ++i)
        _models[i].compute(this->_samples, uni_obs[i], 0.0);
    }
  };







}

#endif
