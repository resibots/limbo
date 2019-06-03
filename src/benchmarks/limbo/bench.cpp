//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#include <chrono>
#include <iostream>

#include <limbo/limbo.hpp>

#include "testfunctions.hpp"

using namespace limbo;

struct Params {
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };
    struct bayes_opt_boptimizer {
#if defined(LIMBO_DEF_HPOPT) || defined(BAYESOPT_DEF_HPOPT)
        BO_PARAM(int, hp_period, 50);
#else
        BO_PARAM(int, hp_period, -1);
#endif
    };
    struct stop_maxiterations {
        BO_PARAM(int, iterations, 190);
    };
    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 1e-10);
    };
    struct kernel_exp : public defaults::kernel_exp {
    };
    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };
    struct acqui_ucb : public defaults::acqui_ucb {
        BO_PARAM(double, alpha, 0.125);
    };
    struct acqui_ei : public defaults::acqui_ei {
    };
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };
    struct mean_constant {
        BO_PARAM(double, constant, 1);
    };
    struct opt_rprop : public defaults::opt_rprop {
        BO_PARAM(double, eps_stop, 1e-6);
    };
    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
    };
#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
        BO_PARAM(int, max_fun_evals, 500);
        BO_PARAM(double, fun_tolerance, 1e-6);
        BO_PARAM(double, xrel_tolerance, 1e-6);
    };
#endif
};

struct DirectParams {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};

struct BobyqaParams {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};
struct BobyqaParams_HP {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};

BO_DECLARE_DYN_PARAM(int, DirectParams::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(int, BobyqaParams::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(int, BobyqaParams_HP::opt_nloptnograd, iterations);

template <typename Optimizer, typename Function>
void benchmark(const std::string& name)
{
    int iters_base = 250;
    DirectParams::opt_nloptnograd::set_iterations(static_cast<int>(iters_base * Function::dim_in() * 0.9));
    BobyqaParams::opt_nloptnograd::set_iterations(iters_base * Function::dim_in() - DirectParams::opt_nloptnograd::iterations());

    BobyqaParams_HP::opt_nloptnograd::set_iterations(10 * Function::dim_in() * Function::dim_in());

    auto t1 = std::chrono::steady_clock::now();
    Optimizer opt;
    Benchmark<Function> target;
    opt.optimize(target);
    auto time_running = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
    std::cout.precision(17);
    std::cout << std::endl;
    auto best = opt.best_observation();
    double accuracy = target.accuracy(best[0]);

    std::cout << name << std::endl;
    std::cout << "Result: " << std::fixed << opt.best_sample().transpose() << " -> " << best << std::endl;
    std::cout << "Smallest difference: " << accuracy << std::endl;
    std::cout << "Time running: " << time_running << "ms" << std::endl
              << std::endl;

    std::ofstream res_file(name + ".dat", std::ios_base::out | std::ios_base::app);
    res_file.precision(17);
    res_file << std::fixed << accuracy << " " << time_running << std::endl;
}

int main()
{
    srand(time(NULL));

// limbo default parameters
#ifdef LIMBO_DEF
    using Opt_t = bayes_opt::BOptimizer<Params>;
#elif defined(LIMBO_DEF_HPOPT)
    using Opt_t = bayes_opt::BOptimizerHPOpt<Params>;

// Bayesopt default parameters
#elif defined(BAYESOPT_DEF_HPOPT)
    using Kernel_t = kernel::SquaredExpARD<Params>;
    using AcquiOpt_t = opt::Chained<Params, opt::NLOptNoGrad<DirectParams, nlopt::GN_DIRECT_L>, opt::NLOptNoGrad<BobyqaParams, nlopt::LN_BOBYQA>>;
    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>>;
    using Mean_t = mean::Constant<Params>;
    using Stat_t = boost::fusion::vector<>;
    using Init_t = init::RandomSampling<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t, model::gp::KernelLFOpt<Params, opt::NLOptNoGrad<BobyqaParams_HP, nlopt::LN_BOBYQA>>>;
    using Acqui_t = acqui::UCB<Params, GP_t>;

    using Opt_t = bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>>;
#elif defined(BAYESOPT_DEF)
    using Kernel_t = kernel::MaternFiveHalves<Params>;
    using AcquiOpt_t = opt::Chained<Params, opt::NLOptNoGrad<DirectParams, nlopt::GN_DIRECT_L>, opt::NLOptNoGrad<BobyqaParams, nlopt::LN_BOBYQA>>;
    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>>;
    using Mean_t = mean::Constant<Params>;
    using Stat_t = boost::fusion::vector<>;
    using Init_t = init::RandomSampling<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t, model::gp::NoLFOpt<Params>>;
    using Acqui_t = acqui::UCB<Params, GP_t>;

    using Opt_t = bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>>;

// benchmark different optimization algorithms
#elif defined(OPT_CMAES)
    using AcquiOpt_t = opt::Cmaes<Params>;
    using Opt_t = bayes_opt::BOptimizer<Params, acquiopt<AcquiOpt_t>>;
#elif defined(OPT_DIRECT)
    using AcquiOpt_t = opt::Chained<Params, opt::NLOptNoGrad<DirectParams, nlopt::GN_DIRECT_L>, opt::NLOptNoGrad<BobyqaParams, nlopt::LN_BOBYQA>>;
    using Opt_t = bayes_opt::BOptimizer<Params, acquiopt<AcquiOpt_t>>;

//benchmark different acquisition functions
#elif defined(ACQ_UCB)
    using GP_t = model::GP<Params>;
    using Acqui_t = acqui::UCB<Params, GP_t>;
    using Opt_t = bayes_opt::BOptimizer<Params, acquifun<Acqui_t>>;
#elif defined(ACQ_EI)
    using GP_t = model::GP<Params>;
    using Acqui_t = acqui::EI<Params, GP_t>;
    using Opt_t = bayes_opt::BOptimizer<Params, acquifun<Acqui_t>>;
#else
#error "Unknown variant in benchmark"
#endif

    benchmark<Opt_t, BraninNormalized>("branin");
    benchmark<Opt_t, Hartmann6>("hartmann6");
    benchmark<Opt_t, Hartmann3>("hartmann3");
    benchmark<Opt_t, Rastrigin>("rastrigin");
    benchmark<Opt_t, Sphere>("sphere");
    benchmark<Opt_t, Ellipsoid>("ellipsoid");
    benchmark<Opt_t, GoldsteinPrice>("goldsteinprice");
    benchmark<Opt_t, SixHumpCamel>("sixhumpcamel");

    return 0;
}
