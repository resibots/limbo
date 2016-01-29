#include <limbo/limbo.hpp>
#include <limbo/bayes_opt/imgpo.hpp>

using namespace limbo;

struct Params {
    struct acqui_ucb : public defaults::acqui_ucb {
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
        BO_PARAM(bool, stats_enabled, false);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 20);
    };
};

struct FuncEval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        // Sphere
        Eigen::Vector2d opt(0.5, 0.5);
        Eigen::VectorXd res(1);
        res(0) = -(x - opt).squaredNorm();
        return res;
    }
};

int main()
{
    typedef kernel::MaternFiveHalfs<Params> Kernel_t;
    typedef mean::Data<Params> Mean_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;
    typedef init::NoInit<Params> Init_t;

    bayes_opt::IMGPO<Params, modelfun<GP_t>, acquifun<Acqui_t>, initfun<Init_t>> opt;

    std::cout << "Optimize using IMGPO" << std::endl;
    opt.optimize(FuncEval());
    std::cout << "best obs: " << opt.best_observation() << " res  " << opt.best_sample().transpose() << std::endl;
    return 0;
}
