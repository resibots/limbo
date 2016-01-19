#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boptimizer

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {

    struct opt_rprop : public defaults::opt_rprop {
    };

    struct opt_gridsearch {
        BO_PARAM(int, bins, 10);
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };

    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.0001);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 50);
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
struct eval3 {
    BOOST_STATIC_CONSTEXPR int dim_in = 3;
    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd v(1);
        Eigen::VectorXd t(3);
        t << 0.1, 0.2, 0.3;
        double y = (x - t).norm();
        v(0) = -y;
        return v;
    }
};

template <typename Params, int obs_size = 1>
struct eval3_blacklist {
    BOOST_STATIC_CONSTEXPR int dim_in = 3;
    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const throw(limbo::EvaluationError)
    {
        tools::rgen_double_t rgen(0, 1);
        if (rgen.rand() < 0.1)
            throw limbo::EvaluationError();
        Eigen::VectorXd v(1);
        Eigen::VectorXd t(3);
        t << 0.1, 0.2, 0.3;
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
        t(0) = 0.1;
        double y = (x - t).norm();
        v(0) = -y;
        return v;
    }
};

BOOST_AUTO_TEST_CASE(test_bo_gp)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::GridSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval3<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_blacklist)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::GridSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval3_blacklist<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_auto)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::GridSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t, model::gp::KernelLFOpt<Params>> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval1<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_auto_mean)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::GridSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::FunctionARD<Params, mean::Data<Params>> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t, model::gp::KernelMeanLFOpt<Params>> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval1<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_mean)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::GridSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::FunctionARD<Params, mean::Data<Params>> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::Observations<Params>> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t, model::gp::MeanLFOpt<Params>> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(eval3<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}
