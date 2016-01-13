#define FUSION_MAX_VECTOR_SIZE 20
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {

    struct kernel_exp {
        BO_PARAM(double, sigma, 1);
    };

    struct kernel_maternthreehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0);
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct acqui_gpucb : public defaults::acqui_gpucb {
    };

    struct opt_gridsearch {
        BO_PARAM(int, bins, 20);
    };
#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#endif
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#endif
    struct opt_rprop : public defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };

    struct init_gridsampling {
        BO_PARAM(int, bins, 5);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 5);
    };

    struct init_randomsamplinggrid {
        BO_PARAM(int, samples, 5);
        BO_PARAM(int, bins, 5);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 20);
    };

    struct stop_maxpredictedvalue {
        BO_PARAM(double, ratio, 2);
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.001);
    };
};

struct MeanEval {
    MeanEval(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP&) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2.5 * x(0);
        res(1) = -4.5 * x(1);
        return res;
    }
};

struct Average {
    typedef double result_type;
    double operator()(const Eigen::VectorXd& x) const
    {
        return (x(0) + x(1)) / 2;
    }
};

struct StateEval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        res(0) = 3 * x(0) + 5;
        res(1) = -5 * x(1) + 2;
        return res;
    }
};

int main()
{
    // clang-format off
    @declarations
    @optimizer.optimize(StateEval());
    @optimizer.best_observation();
    @optimizer.best_sample();
    @optimizer.optimize(StateEval(), Average(), true);
    @optimizer.best_observation(Average());
    @optimizer.best_sample(Average());
    // clang-format on
}
