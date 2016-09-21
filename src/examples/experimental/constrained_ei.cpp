#include <limbo/limbo.hpp>

#include <limbo/experimental/bayes_opt/cboptimizer.hpp>
#include <limbo/experimental/acqui/cei.hpp>

using namespace limbo;

struct Params {
    struct cbayes_opt_boptimizer : public defaults::cbayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.01);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 1);
    };

    struct kernel_exp : public defaults::kernel_exp {
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };

    struct acqui_cei : public defaults::acqui_cei {
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0.0);
    };

#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#elif defined(USE_LIBCMAES)
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif
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
    static constexpr size_t dim_out = 2;

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
    static constexpr size_t nb_constraints = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(4);

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

        // testing the constraints
        (res(0) > 0.4) ? res(2) = 1 : res(2) = 0;
        (res(1) > 0.4) ? res(3) = 1 : res(3) = 0;

        std::cout << res(0) << " " << res(1) << std::endl;
        return res;
    }
};

int main()
{
    tools::par::init();

    // typedef zdt1 func_t;
    // typedef zdt2 func_t;
    // typedef zdt3 func_t;
    typedef mop2 func_t;

    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>>;
    using Stat_t = boost::fusion::vector<stat::Samples<Params>,
        stat::BlSamples<Params>,
        stat::BestObservations<Params>,
        stat::AggregatedObservations<Params>>;
    using Mean_t = mean::Constant<Params>;
    using Kernel_t = kernel::Exp<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t>;

    using Acqui_t = experimental::acqui::CEI<Params, GP_t>;
    using Init_t = init::RandomSampling<Params>;

    experimental::bayes_opt::CBOptimizer<Params,
        modelfun<GP_t>,
        acquifun<Acqui_t>,
        statsfun<Stat_t>,
        initfun<Init_t>,
        stopcrit<Stop_t>>
        opt;

    opt.optimize(func_t());

    return 0;
}
