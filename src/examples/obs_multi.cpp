//#define SHOW_TIMER
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#include <limbo/limbo.hpp>
#include <limbo/inner_opt/cmaes.hpp>

using namespace limbo;

struct Params {
    struct gp_ucb : public defaults::gp_ucb {
    };

    struct cmaes : public defaults::cmaes {
    };

    struct ucb {
        BO_PARAM(float, alpha, 0.1);
    };

    struct kf_maternfivehalfs {
        BO_PARAM(float, sigma, 1);
        BO_PARAM(float, l, 0.2);
    };

    struct boptimizer {
        BO_PARAM(double, noise, 0.001);
        BO_PARAM(int, dump_period, 1);
    };

    struct init {
        BO_PARAM(int, nb_samples, 5);
    };

    struct maxiterations {
        BO_PARAM(int, n_iterations, 20);
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
    typedef kernel_fun::MaternFiveHalfs<Params> Kernel_t;
    typedef mean_fun::Data<Params> Mean_t;
    typedef models::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui_fun::GP_UCB<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>> opt;

    std::cout << "Optimize using  Average aggregator" << std::endl;
    opt.optimize(StateEval(), Average());
    std::cout << "best obs based on Average aggregator: " << opt.best_observation(Average()) << " res  " << opt.best_sample(Average()).transpose() << std::endl;
    std::cout << "best obs based on FirstElem aggregator: " << opt.best_observation(FirstElem()) << " res  " << opt.best_sample(FirstElem()).transpose() << std::endl;
    std::cout << "best obs based on SecondElem aggregator: " << opt.best_observation(SecondElem()) << " res  " << opt.best_sample(SecondElem()).transpose() << std::endl;
    return 0;
}
