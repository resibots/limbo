#ifndef BO_BASE_HPP_
#define BO_BASE_HPP_
#define BOOST_PARAMETER_MAX_ARITY 7
#include <vector>
#include <iostream>
#include <boost/parameter.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/for_each.hpp>

#define BOOST_NO_SCOPED_ENUMS
#include <boost/filesystem.hpp>

#include <Eigen/Core>
#include <limits>

// we need everything to have the defaults
#include <limbo/stop_crit/chain_criteria.hpp>
#include <limbo/stop_crit/max_iterations.hpp>
#include <limbo/stats/acquisitions.hpp>
#include <limbo/misc/sys.hpp>
#include <limbo/kernel_fun/squared_exp_ard.hpp>
#include <limbo/acqui_fun/gp_ucb.hpp>
#include <limbo/mean_fun/data.hpp>
#include <limbo/inner_opt/cmaes.hpp>
#include <limbo/models/gp_auto.hpp>
#include <limbo/init_fun/random_sampling.hpp>

namespace limbo {

    template <typename BO>
    struct RefreshStat_f {
        RefreshStat_f(BO& bo, bool blacklisted)
            : _bo(bo), _blacklisted(blacklisted) {}

        BO& _bo;
        bool _blacklisted;

        template <typename T>
        void operator()(T& x) const { x(_bo, _blacklisted); }
    };

    struct FirstElem {
        typedef double result_type;
        double operator()(const Eigen::VectorXd& x) const
        {
            return x(0);
        }
    };

    // we use optimal named template parameters
    // see:
    // http://www.boost.org/doc/libs/1_55_0/libs/parameter/doc/html/index.html#parameter-enabled-class-templates

    BOOST_PARAMETER_TEMPLATE_KEYWORD(inneropt)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(initfun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(acquifun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(modelfun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(statsfun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(stopcrit)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(obs_type)

    template <typename T, typename std::enable_if<std::is_arithmetic<T>::value, int>::type = 0>
    inline bool is_nan_or_inf(T v)
    {
        return std::isinf(v) || std::isnan(v);
    }

    template <typename T, typename std::enable_if<!std::is_arithmetic<T>::value, int>::type = 0>
    inline bool is_nan_or_inf(const T& v)
    {
        for (int i = 0; i < v.size(); ++i)
            if (std::isinf(v(i)) || std::isnan(v(i)))
                return true;
        return false;
    }

    namespace bayes_opt {

        typedef boost::parameter::parameters<boost::parameter::optional<tag::inneropt>,
            boost::parameter::optional<tag::statsfun>,
            boost::parameter::optional<tag::initfun>,
            boost::parameter::optional<tag::acquifun>,
            boost::parameter::optional<tag::stopcrit>,
            boost::parameter::optional<tag::modelfun>,
            boost::parameter::optional<tag::obs_type>> class_signature;

        template <class Params, class A1 = boost::parameter::void_,
            class A2 = boost::parameter::void_, class A3 = boost::parameter::void_,
            class A4 = boost::parameter::void_, class A5 = boost::parameter::void_,
            class A6 = boost::parameter::void_, class A7 = boost::parameter::void_>
        class BoBase {
        public:
            typedef Params params_t;
            // defaults
            struct defaults {
                typedef init_fun::RandomSampling<Params> init_t; // 1
                typedef inner_opt::Cmaes<Params> inneropt_t; // 2
                typedef kernel_fun::SquaredExpARD<Params> kf_t;
                typedef mean_fun::Data<Params> mean_t;
                typedef models::GPAuto<Params, kf_t, mean_t> model_t; // 3
                // WARNING: you have to specify the acquisition  function
                // if you use a custom model
                typedef acqui_fun::GP_UCB<Params, model_t> acqui_t; // 4
                typedef stats::Acquisitions<Params> stat_t; // 5
                typedef boost::fusion::vector<stop_crit::MaxIterations<Params>> stop_t; // 6
                typedef Eigen::VectorXd obs_t; // 7
            };

            // extract the types
            typedef typename class_signature::bind<A1, A2, A3, A4, A5, A6, A7>::type args;
            typedef typename boost::parameter::binding<args, tag::inneropt, typename defaults::inneropt_t>::type inner_optimization_t;
            typedef typename boost::parameter::binding<args, tag::initfun, typename defaults::init_t>::type init_function_t;
            typedef typename boost::parameter::binding<args, tag::acquifun, typename defaults::acqui_t>::type acquisition_function_t;
            typedef typename boost::parameter::binding<args, tag::modelfun, typename defaults::model_t>::type model_t;
            typedef typename boost::parameter::binding<args, tag::statsfun, typename defaults::stat_t>::type Stat;
            typedef typename boost::parameter::binding<args, tag::stopcrit, typename defaults::stop_t>::type StoppingCriteria;
            typedef typename boost::parameter::binding<args, tag::obs_type, typename defaults::obs_t>::type obs_t;

            typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<StoppingCriteria>, StoppingCriteria, boost::fusion::vector<StoppingCriteria>>::type stopping_criteria_t;
            typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<Stat>, Stat, boost::fusion::vector<Stat>>::type stat_t;

            BoBase() { _make_res_dir(); }

            // disable copy (dangerous and useless)
            BoBase(const BoBase& other) = delete;
            BoBase& operator=(const BoBase& other) = delete;

            bool dump_enabled() const { return Params::boptimizer::dump_period() > 0; }

            const std::string& res_dir() const { return _res_dir; }

            const std::vector<obs_t>& observations() const { return _observations; }

            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

            const std::vector<Eigen::VectorXd>& bl_samples() const { return _bl_samples; }

            int iteration() const { return _iteration; }

            // does not update the model !
            // we don't add NaN and inf observations
            void add_new_sample(const Eigen::VectorXd& s, const obs_t& v)
            {
                if (is_nan_or_inf(v))
                    return;
                _samples.push_back(s);
                _observations.push_back(v);
            }

            void add_new_bl_sample(const Eigen::VectorXd& s) { _bl_samples.push_back(s); }

        protected:
            template <typename F>
            void _init(const F& feval, bool reset = true)
            {
                this->_iteration = 0;
                if (reset) {
                    this->_samples.clear();
                    this->_observations.clear();
                    this->_bl_samples.clear();
                }

                if (this->_samples.empty() && this->_bl_samples.empty())
                    init_function_t()(feval, *this);
            }

            template <typename BO, typename AggregatorFunction>
            bool _pursue(const BO& bo, const AggregatorFunction& afun) const
            {
                stop_crit::ChainCriteria<BO, AggregatorFunction> chain(bo, afun);
                return boost::fusion::accumulate(_stopping_criteria, true, chain);
            }

            template <typename BO>
            void _update_stats(BO& bo, bool blacklisted)
            { // not const, because some stat class
                // modify the optimizer....
                boost::fusion::for_each(_stat, RefreshStat_f<BO>(bo, blacklisted));
            }

            void _make_res_dir()
            {
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
            std::vector<Eigen::VectorXd> _bl_samples;
        };
    }
}

#endif
