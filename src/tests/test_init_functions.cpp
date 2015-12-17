#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE init_functions

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/init.hpp>
#include <limbo/acqui.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>

using namespace limbo;

Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
}

struct Params {
    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };

    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.01);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 0);
    };

    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.25);
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct acqui_gpucb : public defaults::acqui_gpucb {
    };

#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#elif defined(USE_NLOPT)
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif
    struct opt_rprop : public defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
};

struct fit_eval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        double res = 0;
        for (int i = 0; i < x.size(); i++)
            res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
        return make_v1(res);
    }
};

BOOST_AUTO_TEST_CASE(no_init)
{
    std::cout << "NoInit" << std::endl;
    typedef init::NoInit<Params> Init_t;
    typedef bayes_opt::BOptimizer<Params, initfun<Init_t>> Opt_t;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 1);
    BOOST_CHECK(opt.samples().size() == 1);
}

BOOST_AUTO_TEST_CASE(random_sampling)
{
    std::cout << "RandomSampling" << std::endl;
    struct MyParams : public Params {
        struct init_randomsampling {
            BO_PARAM(int, samples, 10);
        };
    };

    typedef init::RandomSampling<MyParams> Init_t;
    typedef bayes_opt::BOptimizer<MyParams, initfun<Init_t>> Opt_t;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 10);
    BOOST_CHECK(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
        }
    }
}

BOOST_AUTO_TEST_CASE(random_sampling_grid)
{
    std::cout << "RandomSamplingGrid" << std::endl;
    struct MyParams : public Params {
        struct init_randomsamplinggrid {
            BO_PARAM(int, samples, 10);
            BO_PARAM(int, bins, 4);
        };
    };

    typedef init::RandomSamplingGrid<MyParams> Init_t;
    typedef bayes_opt::BOptimizer<MyParams, initfun<Init_t>> Opt_t;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 10);
    BOOST_CHECK(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
            BOOST_CHECK(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(grid_sampling)
{
    std::cout << "GridSampling" << std::endl;
    struct MyParams : public Params {
        struct init_gridsampling {
            BO_PARAM(int, bins, 4);
        };
    };

    typedef init::GridSampling<MyParams> Init_t;
    typedef bayes_opt::BOptimizer<MyParams, initfun<Init_t>> Opt_t;

    Opt_t opt;
    opt.optimize(fit_eval());
    std::cout << opt.observations().size() << std::endl;
    BOOST_CHECK(opt.observations().size() == 25);
    BOOST_CHECK(opt.samples().size() == 25);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
            BOOST_CHECK(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
        }
    }
}
