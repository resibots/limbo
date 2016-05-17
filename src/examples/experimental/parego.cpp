#include <limbo/limbo.hpp>
#include <limbo/experimental/bayes_opt/parego.hpp>
#include <limbo/experimental/stat/pareto_front.hpp>
#include <limbo/experimental/stat/hyper_volume.hpp>

using namespace limbo;

struct Params {
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

    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.001);
        BO_PARAM(bool, stats_enabled, true);
    };

    struct SquaredExpARD {
        BO_PARAM(int, k, 0);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 5);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };

    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
    struct model_gp_parego : public experimental::defaults::model_gp_parego {
    };

    struct acqui_gpucb : public defaults::acqui_gpucb {
    };
    struct stat_hyper_volume {
        BO_PARAM_ARRAY(double, ref, 10, 10);
    };
};

// http://www.tik.ee.ethz.ch/sop/download/supplementary/testproblems/zdt2/
// hypervolume : 120,3333 (ref: (11,11))
// see also : https://en.wikipedia.org/wiki/Test_functions_for_optimization
// original is mimimization, objectives in [0,1], we transform it to max
// ref point should be 10, 10 (same hypervolume)
struct zdt2 {
    static constexpr size_t dim_in = 30;
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
        res(0) = -f1;
        res(1) = -f2;
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
    using stat_t = boost::fusion::vector<experimental::stat::ParetoFront<Params>,
        experimental::stat::HyperVolume<Params>,
        stat::ConsoleSummary<Params>>;
    using opt_t = experimental::bayes_opt::Parego<Params, statsfun<stat_t>>;
    opt_t opt;
    opt.optimize(mop2());

    return 0;
}
