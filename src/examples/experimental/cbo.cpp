#include <limbo/limbo.hpp>

#include <limbo/experimental/bayes_opt/cboptimizer.hpp>
#include <limbo/experimental/acqui/eci.hpp>
#include <limbo/experimental/model/mgp.hpp>

using namespace limbo;

struct Params {
    struct bayes_opt_cboptimizer : public defaults::bayes_opt_cboptimizer {
        BO_PARAM(double, noise, 0.01);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };

    struct kernel_exp : public defaults::kernel_exp {
    };

    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };

    struct acqui_eci : public defaults::acqui_eci {
    };

    struct mean_constant {
        BO_PARAM(double, constant, 1.0);
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

struct cosine {
    static constexpr size_t dim_in = 1;
    static constexpr size_t dim_out = 1;
    static constexpr size_t nb_constraints = 1;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        // we _maximize in [0:1]
        Eigen::VectorXd xx = -4.0 + 8.0 * x.array();
        res(0) = std::cos(xx.array()(0));

        // testing the constraints
        std::string feas = "infeasible";
        res(1) = 0;
        if (res(0) < 0.5) {
            res(1) = 1;
            feas = "feasible";
        }
        std::cout << xx(0) << ": " << res(0) << " --> " << feas << std::endl;
        return res;
    }
};

int main()
{
    tools::par::init();
    typedef cosine func_t;

    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>>;
    using Stat_t = boost::fusion::vector<stat::Samples<Params>,
        stat::BestObservations<Params>,
        stat::AggregatedObservations<Params>>;
    using Mean_t = mean::Constant<Params>;
    using Kernel_t = kernel::Exp<Params>;
    using Objective_GP_t = model::GP<Params, Kernel_t, Mean_t>;
    using Constrained_GP_t = model::GP<Params, Kernel_t, Mean_t>;

    using gps_vec_t = boost::fusion::vector<Objective_GP_t, Constrained_GP_t>;
    using MGP_t = experimental::model::MGP<Params, gps_vec_t>;

    using Acqui_t = experimental::acqui::ECI<Params, MGP_t, Constrained_GP_t>;
    using Init_t = init::RandomSampling<Params>;

    experimental::bayes_opt::CBOptimizer<Params,
        modelfun<MGP_t>,
        acquifun<Acqui_t>,
        statsfun<Stat_t>,
        initfun<Init_t>,
        stopcrit<Stop_t>,
        experimental::constraint_modelfun<Constrained_GP_t>>
        opt;

    opt.optimize(func_t());

    size_t n = 0;
    double best = -100;
    for (size_t i = 0; i < opt.samples().size(); i++) {
        Eigen::VectorXd res = func_t()(opt.samples()[i]);
        if (res(0) > best && res(0) < 0.5)
            best = res(0);
        if (res(0) >= 0.5)
            n++;
    }
    std::cout << "Infeasible points tested: " << n << std::endl;
    std::cout << "Best feasible observation: " << best << std::endl;

    return 0;
}
