#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE boptimizer

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {

    struct rprop {
        BO_PARAM(int, n_rprop, 300);
        BO_PARAM(int, rprop_restart, 10);
    };
    struct exhaustive_search {
        BO_PARAM(int, nb_pts, 10);
    };
    struct boptimizer {
        BO_PARAM(double, noise, 0.0001);
        BO_PARAM(int, dump_period, 1);
    };
    struct maxiterations {
        BO_PARAM(int, n_iterations, 30);
    };

    struct kf_maternfivehalfs {
        BO_PARAM(float, sigma, 1);
        BO_PARAM(float, l, 0.4);
    };
    struct ucb {
        BO_PARAM(float, alpha, 0.125);
    };
    struct init {
        BO_PARAM(int, nb_samples, 5);
    };
};

template <typename Params, int obs_size = 1>
struct fit_eval_map {

    BOOST_STATIC_CONSTEXPR int dim_in = 3;

    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;
    fit_eval_map() {}

    Eigen::VectorXd operator()(Eigen::VectorXd x) const
    {
        Eigen::VectorXd v(1);
        Eigen::VectorXd t(3);
        t(0) = 0.1;
        t(1) = 0.2;
        t(2) = 0.3; //t(3) = 0.4; t(4) = 0.5; t(5) = 0.6;
        double y = (x - t).norm();
        v(0) = -y;
        return v;
    }
};

BOOST_AUTO_TEST_CASE(test_bo_gp)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::ExhaustiveSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Acquisitions<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    opt.optimize(fit_eval_map<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_auto)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::ExhaustiveSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::Data<Params> Mean_t;
    typedef boost::fusion::vector<stat::Acquisitions<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;
    typedef opt::impl::GPKernelLFOpt<Params> opt_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>, optfun<opt_t>> opt;
    opt.optimize(fit_eval_map<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_auto_mean)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::ExhaustiveSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::FunctionARD<Params, mean::Data<Params>> Mean_t;
    typedef boost::fusion::vector<stat::Acquisitions<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;
    typedef opt::impl::GPKernelMeanLFOpt<Params> opt_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>, optfun<opt_t>> opt;
    opt.optimize(fit_eval_map<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}

BOOST_AUTO_TEST_CASE(test_bo_gp_mean)
{
    using namespace limbo;

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::ExhaustiveSearch<Params> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean::FunctionARD<Params, mean::Data<Params>> Mean_t;
    typedef boost::fusion::vector<stat::Acquisitions<Params>> Stat_t;
    typedef init::NoInit<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;
    typedef opt::impl::GPMeanLFOpt<Params> opt_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>, optfun<opt_t>> opt;
    opt.optimize(fit_eval_map<Params>());

    BOOST_CHECK_CLOSE(opt.best_sample()(0), 0.1, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(1), 0.2, 0.000001);
    BOOST_CHECK_CLOSE(opt.best_sample()(2), 0.3, 0.000001);
}
