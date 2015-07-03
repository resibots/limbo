#ifndef BO_BASE_HPP_
#define BO_BASE_HPP_
#define BOOST_PARAMETER_MAX_ARITY 7
#include <vector>
#include <iostream>
#include <boost/parameter.hpp>
#include <boost/progress.hpp>
#include <boost/fusion/algorithm.hpp>
#include <boost/fusion/algorithm/iteration/accumulate.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#define BOOST_NO_SCOPED_ENUMS
#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <Eigen/LU>
#include <limits>


// we need everything to have the defaults
#include "macros.hpp"
#include "stopping_criterion.hpp"
#include "stat.hpp"
#include "sys.hpp"
#include "rand.hpp"
#include "kernel_functions.hpp"
#include "acquisition_functions.hpp"
#include "mean_functions.hpp"
#include "inner_optimization.hpp"
#include "inner_cmaes.hpp"
#include "gp.hpp"
#include "gp_auto.hpp"
#include "init_functions.hpp"


namespace limbo {


  template<typename BO>
  struct RefreshStat_f {
    RefreshStat_f(BO &bo) : _bo(bo) {// not const, because some stat class modify the optimizer....
    }
    BO& _bo;
    template<typename T>
    void operator() (T & x) const {
      x(_bo);
    }
  };

  // we use optimal named template parameters
  // see: http://www.boost.org/doc/libs/1_55_0/libs/parameter/doc/html/index.html#parameter-enabled-class-templates

  BOOST_PARAMETER_TEMPLATE_KEYWORD(inneropt_fun)
  BOOST_PARAMETER_TEMPLATE_KEYWORD(init_fun)
  BOOST_PARAMETER_TEMPLATE_KEYWORD(acq_fun)
  BOOST_PARAMETER_TEMPLATE_KEYWORD(model_fun)
  BOOST_PARAMETER_TEMPLATE_KEYWORD(stat_fun)
  BOOST_PARAMETER_TEMPLATE_KEYWORD(stop_fun)
  BOOST_PARAMETER_TEMPLATE_KEYWORD(obs_type)

  template<typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
  inline bool is_nan_or_inf(T v) {
    return std::isinf(v) || std::isnan(v);
  }

  template < typename T, typename std::enable_if < !std::is_arithmetic<T>::value, int >::type = 0 >
  inline bool is_nan_or_inf(const T& v) {
    for (int i = 0; i < v.size(); ++i)
      if (std::isinf(v(i)) || std::isnan(v(i)))
        return true;
    return false;
  }


  typedef boost::parameter::parameters <
  boost::parameter::optional<tag::inneropt_fun>
  , boost::parameter::optional<tag::stat_fun>
  , boost::parameter::optional<tag::init_fun>
  , boost::parameter::optional<tag::acq_fun>
  , boost::parameter::optional<tag::stop_fun>
  , boost::parameter::optional<tag::model_fun>
  , boost::parameter::optional<tag::obs_type>
  > class_signature;

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
  class BoBase {
   public:
    typedef Params params_t;
    //defaults
    struct defaults {
      typedef init_functions::RandomSampling<Params> init_t; // 1
      typedef inner_optimization::Cmaes<Params> inneropt_t;  // 2
      typedef kernel_functions::SquaredExpARD<Params> kf_t;
      typedef mean_functions::MeanData<Params> mean_t;
      typedef model::GPAuto<Params, kf_t, mean_t> model_t; //3
      // WARNING: you have to specify the acquisition  function
      // if you use a custom model
      typedef acquisition_functions::GP_UCB<Params, model_t> acqui_t; //4
      typedef stat::Acquisitions<Params> stat_t; //5
      typedef boost::fusion::vector<stopping_criterion::MaxIterations<Params> > stop_t; //6
      typedef double obs_t; //7
    };

    // extract the types
    typedef typename class_signature::bind<A1, A2, A3, A4, A5, A6, A7>::type args;
    typedef typename boost::parameter::binding<args, tag::inneropt_fun, typename defaults::inneropt_t>::type inner_optimization_t;
    typedef typename boost::parameter::binding<args, tag::init_fun, typename defaults::init_t>::type init_function_t;
    typedef typename boost::parameter::binding<args, tag::acq_fun, typename defaults::acqui_t>::type acquisition_function_t;
    typedef typename boost::parameter::binding<args, tag::model_fun, typename defaults::model_t>::type model_t;
    typedef typename boost::parameter::binding<args, tag::stat_fun, typename defaults::stat_t>::type Stat;
    typedef typename boost::parameter::binding<args, tag::stop_fun, typename defaults::stop_t>::type StoppingCriteria;
    typedef typename boost::parameter::binding<args, tag::obs_type, typename defaults::obs_t>::type obs_t;

    typedef typename
    boost::mpl::if_<boost::fusion::traits::is_sequence<StoppingCriteria>,
          StoppingCriteria,
          boost::fusion::vector<StoppingCriteria> >::type stopping_criteria_t;
    typedef typename
    boost::mpl::if_<boost::fusion::traits::is_sequence<Stat>,
          Stat,
          boost::fusion::vector<Stat> >::type stat_t;

    BoBase() {
      _make_res_dir();
    }

    // disable copy (dangerous and useless)
    BoBase(const BoBase& other) = delete;
    BoBase& operator=(const BoBase& other) = delete;

    bool dump_enabled() const {
      return Params::boptimizer::dump_period() > 0;
    }
    const std::string& res_dir() const {
      return _res_dir;
    }
    const std::vector<obs_t>& observations() const {
      return _observations;
    }
    const std::vector<Eigen::VectorXd>& samples() const {
      return _samples;
    }
    int iteration() const {
      return _iteration;
    }



    // does not update the model !
    // we don't add NaN and inf observations
    void add_new_sample(const Eigen::VectorXd& s, const obs_t& v) {
      if (is_nan_or_inf(v))
        return;
      _samples.push_back(s);
      _observations.push_back(v);
    }
   protected:
    template<typename F>
    void _init(const F& feval, bool reset = true) {
      this->_iteration = 0;
      if (reset) {
        this->_samples.clear();
        this->_observations.clear();
      }

      if (this->_samples.empty())
        init_function_t()(feval, *this);
    }
    template<typename BO>
    bool _pursue(const BO& bo) const {
      stopping_criterion::ChainCriteria<BO> chain(bo);
      return boost::fusion::accumulate(_stopping_criteria, true, chain);
    }
    template<typename BO>
    void _update_stats(BO& bo) { // not const, because some stat class modify the optimizer....
      boost::fusion::for_each(_stat, RefreshStat_f<BO>(bo));
    }
    void _make_res_dir() {
      if (Params::boptimizer::dump_period() <= 0)
        return;
      _res_dir = misc::hostname() + "_" + misc::date() + "_" + misc::getpid();
      boost::filesystem::path my_path(_res_dir);
      boost::filesystem::create_directory(my_path);
    }

    std::string _res_dir;
    int _iteration;
    stopping_criteria_t _stopping_criteria;
    stat_t _stat;

    std::vector<obs_t> _observations;
    std::vector<Eigen::VectorXd> _samples;
  };







}

#endif
