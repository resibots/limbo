//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//| 
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
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
#include <iostream>
#include <chrono>

#include <limbo/limbo.hpp>

#include "testfunctions.hpp"

using namespace limbo;

struct Params {
    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };
    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 1e-10);
        BO_PARAM(int, hp_period, -1);
    };
    struct stop_maxiterations {
        BO_PARAM(int, iterations, 190);
    };
    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };
    struct acqui_ucb {
        BO_PARAM(double, alpha, 0.125);
    };
    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };
    struct mean_constant {
        BO_PARAM(double, constant, 1);
    };
};

struct DirectParams {
    struct opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};

struct BobyqaParams {
    struct opt_nloptnograd {
        BO_DYN_PARAM(int, iterations);
    };
};

BO_DECLARE_DYN_PARAM(int, DirectParams::opt_nloptnograd, iterations);
BO_DECLARE_DYN_PARAM(int, BobyqaParams::opt_nloptnograd, iterations);

template <typename Optimizer, typename Function>
void benchmark(const std::string& name)
{
    int iters_base = 250;
    DirectParams::opt_nloptnograd::set_iterations(static_cast<int>(iters_base * Function::dim_in * 0.9));
    BobyqaParams::opt_nloptnograd::set_iterations(iters_base * Function::dim_in - DirectParams::opt_nloptnograd::iterations());

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

    std::ofstream res_file(name + ".dat", std::ios_base::out | std::ios_base::app);
    res_file.precision(17);
    res_file << std::fixed << accuracy << " " << time_running << std::endl;
}

int main()
{
    srand(time(NULL));

    typedef kernel::MaternFiveHalves<Params> Kernel_t;
    typedef opt::Chained<Params, opt::NLOptNoGrad<DirectParams, nlopt::GN_DIRECT_L>, opt::NLOptNoGrad<BobyqaParams, nlopt::LN_BOBYQA>> AcquiOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    // typedef mean_functions::MeanFunctionARD<Params, mean_functions::MeanData<Params>> Mean_t;
    typedef mean::Constant<Params> Mean_t;
    typedef boost::fusion::vector<> Stat_t;
    typedef init::RandomSampling<Params> Init_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;

    typedef bayes_opt::BOptimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<AcquiOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> Opt_t;

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
