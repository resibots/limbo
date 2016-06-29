#include <limbo/limbo.hpp>
#include <limbo/experimental/bayes_opt/parego.hpp>
#include <limbo/experimental/bayes_opt/nsbo.hpp>
#include <limbo/experimental/bayes_opt/ehvi.hpp>
#include <limbo/experimental/stat/pareto_benchmark.hpp>

using namespace limbo;

struct Params {
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.01);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };

    struct kernel_exp : public defaults::kernel_exp {
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 30);
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct acqui_gpucb : public defaults::acqui_gpucb {
    };

    struct opt_cmaes : public defaults::opt_cmaes {
    };

    struct mean_constant : public defaults::mean_constant {
    };

    struct bayes_opt_ehvi {
        BO_PARAM(double, x_ref, -11);
        BO_PARAM(double, y_ref, -11);
    };
};

#ifdef DIM6
#define ZDT_DIM 6
#elif defined(DIM2)
#define ZDT_DIM 2
#else
#define ZDT_DIM 30
#endif

struct zdt1 {
    static constexpr size_t dim_in = ZDT_DIM;
    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        double f1 = x(0);
        double g = 1.0;
        for (int i = 1; i < x.size(); ++i)
            g += 9.0 / (x.size() - 1) * x(i);
        double h = 1.0f - sqrtf(f1 / g);
        double f2 = g * h;
        res(0) = 1.0 - f1;
        res(1) = 1.0 - f2;
        return res;
    }
};

struct zdt2 {
    static constexpr size_t dim_in = ZDT_DIM;
    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        double f1 = x(0);
        double g = 1.0;
        for (int i = 1; i < x.size(); ++i)
            g += 9.0 / (x.size() - 1) * x(i);
        double h = 1.0f - pow((f1 / g), 2.0);
        double f2 = g * h;
        res(0) = 1.0 - f1;
        res(1) = 1.0 - f2;
        return res;
    }
};

struct zdt3 {
    static constexpr size_t dim_in = ZDT_DIM;
    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        double f1 = x(0);
        double g = 1.0;
        for (int i = 1; i < x.size(); ++i)
            g += 9.0 / (x.size() - 1) * x(i);
        double h = 1.0f - sqrtf(f1 / g) - f1 / g * sin(10 * M_PI * f1);
        double f2 = g * h;
        res(0) = 1.0 - f1;
        res(1) = 1.0 - f2;
        return res;
    }
};

struct mop2 {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        // scale to [-2, 2]
        Eigen::VectorXd xx = (x * 4.0).array() - 2.0;
        // f1, f2
        Eigen::VectorXd v1 = (xx.array() - 1.0 / sqrt(xx.size())).array().square();
        Eigen::VectorXd v2 = (xx.array() + 1.0 / sqrt(xx.size())).array().square();
        double f1 = 1.0 - exp(-v1.sum());
        double f2 = 1.0 - exp(-v2.sum());
        // we _maximize in [0:1]
        res(0) = 1 - f1;
        res(1) = 1 - f2;
        return res;
    }
};

int main()
{
    tools::par::init();

#ifdef ZDT1
    typedef zdt1 func_t;
#elif defined ZDT2
    typedef zdt2 func_t;
#elif defined ZDT3
    typedef zdt3 func_t;
#else
    typedef mop2 func_t;
#endif

    using stat_t = boost::fusion::vector<experimental::stat::ParetoBenchmark<func_t>>;

#ifdef PAREGO
    Parego<Params, statsfun<stat_t>> opt;
#elif defined(NSBO)
    Nsbo<Params, statsfun<stat_t>> opt;
#else
    experimental::bayes_opt::Ehvi<Params, statsfun<stat_t>> opt;
#endif
    opt.optimize(func_t());

    return 0;
}
