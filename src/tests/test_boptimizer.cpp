#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_boptimizer

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {

    struct opt_rprop : public defaults::opt_rprop {
    };

#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#elif defined(USE_NLOPT)
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#endif

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };

    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.0001);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 75);
    };

    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };

    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.4);
    };

    struct acqui_ucb {
        BO_PARAM(double, alpha, 0.125);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 20);
    };

    struct opt_parallelrepeater : defaults::opt_parallelrepeater {
    };
};

template <typename Params, int obs_size = 1>
struct eval2 {
    BOOST_STATIC_CONSTEXPR int dim_in = 2;
    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd v(1);
        Eigen::VectorXd t(2);
        t << 0.25, 0.75;
        double y = (x - t).norm();
        v(0) = -y;
        return v;
    }
};

template <typename Params, int obs_size = 1>
struct eval2_blacklist {
    BOOST_STATIC_CONSTEXPR int dim_in = 2;
    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const throw(limbo::EvaluationError)
    {
        tools::rgen_double_t rgen(0, 1);
        if (rgen.rand() < 0.05)
            throw limbo::EvaluationError();
        Eigen::VectorXd v(1);
        Eigen::VectorXd t(2);
        t << 0.25, 0.75;
        double y = (x - t).norm();
        v(0) = -y;
        return v;
    }
};

template <typename Params, int obs_size = 1>
struct eval1 {
    BOOST_STATIC_CONSTEXPR int dim_in = 1;
    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd v(1);
        Eigen::VectorXd t(1);
        t(0) = 0.25;
        double y = (x - t).norm();
        v(0) = -y;
        return v;
    }
};

BOOST_AUTO_TEST_CASE(test_bo_inheritance)
{
    using namespace limbo;

    struct Parameters {
        struct stop_maxiterations {
            BO_PARAM(int, iterations, 1);
        };
    };

    typedef kernel::SquaredExpARD<Params> Kernel_t;
#ifdef USE_LIBCMAES
    typedef opt::Cmaes<Params> AcquiOpt_t;
#else
    typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND> AcquiOpt_t;
#endif
    typedef boost::fusion::vector<stop::MaxIterations<Parameters>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval2<Params>());

    BOOST_CHECK(opt.total_iterations() == 1);
}

BOOST_AUTO_TEST_CASE(test_bo_gp)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
#ifdef USE_LIBCMAES
    typedef opt::Cmaes<Params> AcquiOpt_t;
#else
    typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND> AcquiOpt_t;
#endif
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval2<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.25, 10);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.75, 10);
}

BOOST_AUTO_TEST_CASE(test_bo_blacklist)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
#ifdef USE_LIBCMAES
    typedef opt::Cmaes<Params> AcquiOpt_t;
#else
    typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND> AcquiOpt_t;
#endif
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval2_blacklist<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.25, 10);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.75, 10);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_auto)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
#ifdef USE_LIBCMAES
    typedef opt::Cmaes<Params> AcquiOpt_t;
#else
    typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND> AcquiOpt_t;
#endif
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t, model::gp::KernelLFOpt<Params>> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval1<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.25, 10);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_mean)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
#ifdef USE_LIBCMAES
    typedef opt::Cmaes<Params> AcquiOpt_t;
#else
    typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L_RAND> AcquiOpt_t;
#endif
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::FunctionARD<Params, mean::Data<Params>> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t, model::gp::MeanLFOpt<Params>> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval2<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.25, 10);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.75, 10);
}
