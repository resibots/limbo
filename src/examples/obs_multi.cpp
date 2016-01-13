#include <limbo/tools/macros.hpp>
#include <limbo/kernel/matern_five_halfs.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/acqui/gp_ucb.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>

using namespace limbo;

struct Params {
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
    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.001);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 5);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 20);
    };
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

struct Average {
    typedef double result_type;
    double operator()(const Eigen::VectorXd& x) const
    {
        return x.sum() / x.size();
    }
};

struct SecondElem {
    typedef double result_type;
    double operator()(const Eigen::VectorXd& x) const
    {
        return x(1);
    }
};

int main()
{
    typedef kernel::MaternFiveHalfs<Params> Kernel_t;
    typedef mean::Data<Params> Mean_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::GP_UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>> opt;

    std::cout << "Optimize using  Average aggregator" << std::endl;
    opt.optimize(StateEval(), Average());
    std::cout << "best obs based on Average aggregator: " << opt.best_observation(Average()) << " res  " << opt.best_sample(Average()).transpose() << std::endl;
    std::cout << "best obs based on FirstElem aggregator: " << opt.best_observation(FirstElem()) << " res  " << opt.best_sample(FirstElem()).transpose() << std::endl;
    std::cout << "best obs based on SecondElem aggregator: " << opt.best_observation(SecondElem()) << " res  " << opt.best_sample(SecondElem()).transpose() << std::endl;
    return 0;
}
