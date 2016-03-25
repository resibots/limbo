#ifndef LIMBO_BAYES_OPT_BO_BASE_HPP
#define LIMBO_BAYES_OPT_BO_BASE_HPP

#include <vector>
#include <iostream>
#include <limits>
#include <exception>

#include <boost/parameter.hpp>
#include <boost/fusion/include/vector.hpp>
#include <boost/fusion/include/accumulate.hpp>
#include <boost/fusion/include/for_each.hpp>
#define BOOST_NO_SCOPED_ENUMS
#include <boost/filesystem.hpp>

#include <Eigen/Core>

// we need everything to have the defaults
#include <limbo/tools/macros.hpp>
#include <limbo/stop/chain_criteria.hpp>
#include <limbo/stop/max_iterations.hpp>
#include <limbo/stat/samples.hpp>
#include <limbo/stat/aggregated_observations.hpp>
#include <limbo/stat/console_summary.hpp>
#include <limbo/tools/sys.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/acqui/gp_ucb.hpp>
#include <limbo/mean/data.hpp>
#ifdef USE_LIBCMAES
#include <limbo/opt/cmaes.hpp>
#elif defined USE_NLOPT
#include <limbo/opt/nlopt_no_grad.hpp>
#else
#include <limbo/opt/grid_search.hpp>
#endif
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/init/random_sampling.hpp>

namespace limbo {
    namespace defaults {
        struct bayes_opt_bobase {
            BO_PARAM(bool, stats_enabled, true);
        };
    }
    template <typename BO, typename AggregatorFunction>
    struct RefreshStat_f {
        RefreshStat_f(BO& bo, const AggregatorFunction& afun, bool blacklisted)
            : _bo(bo), _afun(afun), _blacklisted(blacklisted) {}

        BO& _bo;
        const AggregatorFunction& _afun;
        bool _blacklisted;

        template <typename T>
        void operator()(T& x) const { x(_bo, _afun, _blacklisted); }
    };

    struct FirstElem {
        typedef double result_type;
        double operator()(const Eigen::VectorXd& x) const
        {
            return x(0);
        }
    };
    class EvaluationError : public std::exception {
    };

    // we use optimal named template parameters
    // see:
    // http://www.boost.org/doc/libs/1_55_0/libs/parameter/doc/html/index.html#parameter-enabled-class-templates

    BOOST_PARAMETER_TEMPLATE_KEYWORD(initfun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(acquifun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(modelfun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(statsfun)
    BOOST_PARAMETER_TEMPLATE_KEYWORD(stopcrit)

    namespace bayes_opt {

        typedef boost::parameter::parameters<boost::parameter::optional<tag::statsfun>,
            boost::parameter::optional<tag::initfun>,
            boost::parameter::optional<tag::acquifun>,
            boost::parameter::optional<tag::stopcrit>,
            boost::parameter::optional<tag::modelfun>> class_signature;

        // clang-format off
        template <class Params,
          class A1 = boost::parameter::void_,
          class A2 = boost::parameter::void_,
          class A3 = boost::parameter::void_,
          class A4 = boost::parameter::void_,
          class A5 = boost::parameter::void_>
        // clang-format on
        /** Base class for Bayesian optimizers

        **Parameters**
        - ``bool Params::bayes_opt_bobase::stats_enabled``: activate / deactivate the statistics

        This class is templated by several types with default values (thanks to boost::parameters).

        \rst

        +----------------+---------+---------+---------------+
        |type            |typedef  | argument| default       |
        +================+=========+=========+===============+
        |init. func.     |init_t   | initfun | RandomSampling|
        +----------------+---------+---------+---------------+
        |model           |model_t  | modelfun| GP<...>       |
        +----------------+---------+---------+---------------+
        |acquisition fun.|aqui_t   | acquifun| GP_UCB        |
        +----------------+---------+---------+---------------+
        |statistics      | stat_t  | statfun | see below     |
        +----------------+---------+---------+---------------+
        |stopping crit.  | stop_t  | stopcrit| MaxIterations |
        +----------------+---------+---------+---------------+

        For GP, the default value is: ``model::GP<Params, kf_t, mean_t, opt_t>>``,
        - with ``kf_t = kernel::SquaredExpARD<Params>``
        - with ``mean_t = mean::Data<Params>``
        - with ``opt_t = model::gp::KernelLFOpt<Params>``

         (meaning: kernel with automatic relevance determination and mean equals to the mean of the input data, that is, center the data automatically)

        For Statistics, the default value is: ``boost::fusion::vector<stat::Samples<Params>, stat::AggregatedObservations<Params>, stat::ConsoleSummary<Params>>``

        \endrst


         Example of customization:

         - typedef kernel::MaternFiveHalfs<Params> Kernel_t;
         - typedef mean::Data<Params> Mean_t;
         - typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
         - typedef acqui::UCB<Params, GP_t> Acqui_t;
         - bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>> opt;

         @see limbo::bayes_opt::Boptimizer
        */
        class BoBase {
        public:
            typedef Params params_t;
            // defaults
            struct defaults {
                typedef init::RandomSampling<Params> init_t; // 1

                typedef kernel::SquaredExpARD<Params> kf_t;
                typedef mean::Data<Params> mean_t;
                typedef model::GP<Params, kf_t, mean_t, model::gp::KernelLFOpt<Params>> model_t; // 2
                // WARNING: you have to specify the acquisition  function
                // if you use a custom model
                typedef acqui::GP_UCB<Params, model_t> acqui_t; // 3
                typedef boost::fusion::vector<stat::Samples<Params>, stat::AggregatedObservations<Params>, stat::ConsoleSummary<Params>> stat_t; // 4
                typedef boost::fusion::vector<stop::MaxIterations<Params>> stop_t; // 5
            };

            // extract the types
            typedef typename class_signature::bind<A1, A2, A3, A4, A5>::type args;
            typedef typename boost::parameter::binding<args, tag::initfun, typename defaults::init_t>::type init_function_t;
            typedef typename boost::parameter::binding<args, tag::acquifun, typename defaults::acqui_t>::type acquisition_function_t;
            typedef typename boost::parameter::binding<args, tag::modelfun, typename defaults::model_t>::type model_t;
            typedef typename boost::parameter::binding<args, tag::statsfun, typename defaults::stat_t>::type Stat;
            typedef typename boost::parameter::binding<args, tag::stopcrit, typename defaults::stop_t>::type StoppingCriteria;

            typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<StoppingCriteria>, StoppingCriteria, boost::fusion::vector<StoppingCriteria>>::type stopping_criteria_t;
            typedef typename boost::mpl::if_<boost::fusion::traits::is_sequence<Stat>, Stat, boost::fusion::vector<Stat>>::type stat_t;

            /// default constructor
            BoBase() : _total_iterations(0) { _make_res_dir(); }

            /// copy is disabled (dangerous and useless)
            BoBase(const BoBase& other) = delete;
            /// copy is disabled (dangerous and useless)
            BoBase& operator=(const BoBase& other) = delete;

            /// @return true if the statitics are enabled (they can be disabled to avoid dumping data, e.g. for unit tests)
            bool stats_enabled() const { return Params::bayes_opt_bobase::stats_enabled(); }

            /// @return the name of the directory in which results (statistics) are written
            const std::string& res_dir() const { return _res_dir; }

            ///@return the vector of points of observations (observations can be multi-dimensional, hence the VectorXd) -- f(x)
            const std::vector<Eigen::VectorXd>& observations() const { return _observations; }

            ///@return list of the points that have been evaluated so far (x)
            const std::vector<Eigen::VectorXd>& samples() const { return _samples; }

            ///@return list of blacklisted points
            const std::vector<Eigen::VectorXd>& bl_samples() const { return _bl_samples; }

            ///@return the current iteration number
            int current_iteration() const { return _current_iteration; }

            int total_iterations() const { return _total_iterations; }

            /// Add a new sample / observation pair
            /// - does not update the model!
            /// - we don't add NaN and inf observations
            void add_new_sample(const Eigen::VectorXd& s, const Eigen::VectorXd& v)
            {
                if (tools::is_nan_or_inf(v))
                    throw EvaluationError();
                _samples.push_back(s);
                _observations.push_back(v);
            }

            /// Add a new blacklisted sample
            void add_new_bl_sample(const Eigen::VectorXd& s) { _bl_samples.push_back(s); }

            /// Evaluate a sample and add the result to the 'database' (sample / observations vectors)
            template <typename StateFunction>
            bool eval_and_add(const StateFunction& seval, const Eigen::VectorXd& sample)
            {
                try {
                    this->add_new_sample(sample, seval(sample));
                }
                catch (const EvaluationError& e) {
                    this->add_new_bl_sample(sample);
                    return false;
                }

                return true;
            }

        protected:
            template <typename StateFunction, typename AggregatorFunction>
            void _init(const StateFunction& seval, const AggregatorFunction& afun, bool reset = true)
            {
                this->_current_iteration = 0;
                if (reset) {
                    this->_total_iterations = 0;
                    this->_samples.clear();
                    this->_observations.clear();
                    this->_bl_samples.clear();
                }

                if (this->_total_iterations == 0)
                    init_function_t()(seval, afun, *this);
            }

            template <typename BO, typename AggregatorFunction>
            bool _stop(const BO& bo, const AggregatorFunction& afun) const
            {
                stop::ChainCriteria<BO, AggregatorFunction> chain(bo, afun);
                return boost::fusion::accumulate(_stopping_criteria, false, chain);
            }

            template <typename BO, typename AggregatorFunction>
            void _update_stats(BO& bo, const AggregatorFunction& afun, bool blacklisted)
            { // not const, because some stat class
                // modify the optimizer....
                boost::fusion::for_each(_stat, RefreshStat_f<BO, AggregatorFunction>(bo, afun, blacklisted));
            }

            void _make_res_dir()
            {
                if (!Params::bayes_opt_bobase::stats_enabled())
                    return;
                _res_dir = tools::hostname() + "_" + tools::date() + "_" + tools::getpid();
                boost::filesystem::path my_path(_res_dir);
                boost::filesystem::create_directory(my_path);
            }

            std::string _res_dir;
            int _current_iteration;
            int _total_iterations;
            stopping_criteria_t _stopping_criteria;
            stat_t _stat;

            std::vector<Eigen::VectorXd> _observations;
            std::vector<Eigen::VectorXd> _samples;
            std::vector<Eigen::VectorXd> _bl_samples;
        };
    }
}

#endif
