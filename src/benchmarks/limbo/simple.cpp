#include <iostream>
#include <chrono>

#include <limbo/limbo.hpp>

#include "testfunctions.hpp"

using namespace limbo;

struct Params {
  struct bayes_opt_bobase {
    BO_PARAM(bool, stats_enabled, true);
  };
    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 1e-10);
        BO_PARAM(int, dump_period, -1);
    };
    struct stop_maxiterations {
        BO_PARAM(int, iterations, 190);
    };

    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 1);
    };
    struct acqui_ucb {
        BO_PARAM(double, alpha, 0.125);
    };
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
    struct opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
        BO_PARAM(double, epsilon, 1e-8);
    };

  struct mean_constant {
    BO_PARAM(double, constant, 1);
  };
};

BO_DECLARE_DYN_PARAM(int, Params::opt_nloptnograd, iterations);

template <typename Params>
struct StaticInit {
    template <typename StateFunction, typename AggregatorFunction, typename Opt>
    void operator()(const StateFunction& seval, const AggregatorFunction&, Opt& opt) const
    {
        Eigen::Vector2d x;
        x(0) = -0.055497063038156969690135505676;
        x(1) = -0.351795677434771201331992637257;
        opt.add_new_sample(x, seval(x));
        x(0) = 0.240911582092300627645879219810;
        x(1) = -0.351795677434771201331992637257;
        opt.add_new_sample(x, seval(x));
        x(0) = -0.351795677434771201331992637257;
        x(1) = -0.343720034134076580133895794673;
        opt.add_new_sample(x, seval(x));
        x(0) = 0.055387032304313603995977911771;
        x(1) = -0.076908787408790806346486820340;
        opt.add_new_sample(x, seval(x));
        x(0) = 0.351795677434771201331992637257;
        x(1) = -0.076908787408790806346486820339;
        opt.add_new_sample(x, seval(x));
        x(0) = -0.351795677434771201331992637257;
        x(1) = -0.047311389003618982797881069187;
        opt.add_new_sample(x, seval(x));
        x(0) = -0.132623434010086730621834242216;
        x(1) = 0.152242144215576109267055784036;
        opt.add_new_sample(x, seval(x));
        x(0) = 0.351795677434771201331992637257;
        x(1) = 0.219499857721666790989527905147;
        opt.add_new_sample(x, seval(x));
        x(0) = -0.351795677434771201331992637257;
        x(1) = 0.351795677434771201331992637257;
        opt.add_new_sample(x, seval(x));
        x(0) = 0.086548809414597740088324152827;
        x(1) = 0.351795677434771201331992637257;
        opt.add_new_sample(x, seval(x));
    }
};

template <typename Optimizer, typename Function>
void benchmark(const std::string& name)
{
    int iters_base = 250;
    Params::opt_nloptnograd::set_iterations(iters_base * Function::dim_in);
    auto t1 = std::chrono::steady_clock::now();
    Optimizer opt;
    Benchmark<Function> target;
    opt.optimize(target);
    auto time_running = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
    std::cout.precision(17);
    std::cout << std::endl;
    auto best = opt.best_observation();
    double accuracy = target.accuracy(best);

    std::cout << name << std::endl;
    std::cout << "Result: " << std::fixed << opt.best_sample().transpose() << " -> " << best << std::endl;
    std::cout << "Smallest difference: " << accuracy << std::endl;
    std::cout << "Time running: " << time_running << "ms" << std::endl
              << std::endl;

    std::ofstream res_file(name + ".dat",std::ios_base::out | std::ios_base::app);
    res_file.precision(17);
    res_file << std::fixed << accuracy << " " << time_running << std::endl;
}

int main()
{
  srand(time(NULL));

    typedef kernel::MaternFiveHalfs<Params> Kernel_t;
    typedef opt::NLOptNoGrad<Params, nlopt::GN_DIRECT_L> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Constant<Params> Mean_t;
    typedef boost::fusion::vector<> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    typedef bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> Opt_t;

    benchmark<Opt_t, BraninNormalized>("branin");
    benchmark<Opt_t, Hartman6>("hartman6");
    benchmark<Opt_t, Hartman3>("hartman3");
    benchmark<Opt_t, Rastrigin>("rastrigin");
    benchmark<Opt_t, Sphere>("sphere");
    benchmark<Opt_t, Ellipsoid>("ellipsoid");
    benchmark<Opt_t, GoldenPrice>("goldenprice");
    benchmark<Opt_t, SixHumpCamel>("sixhumpcamel");

    return 0;
}
