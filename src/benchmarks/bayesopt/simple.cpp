#define SKIP_TRICKS
#include "testfunctions.hpp"
#include <chrono>
#include <fstream>
#include <string>

template <typename Function>
void benchmark(const bopt_params& par, const std::string& name)
{
    auto t1 = std::chrono::steady_clock::now();
    Benchmark<Function> benchmark(par);
    vectord result(Function::dim_in);
    benchmark.optimize(result);
    auto time_running = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t1).count();
    std::cout.precision(17);
    std::cout << std::endl;
    auto best = benchmark.evaluateSample(result);
    double accuracy = benchmark.accuracy(best);

    std::cout << name << std::endl;
    std::cout << "Result: " << std::fixed << result << " -> " << best << std::endl;
    std::cout << "Smallest difference: " << accuracy << std::endl;
    std::cout << "Time running: " << time_running << "ms" << std::endl
              << std::endl;

    std::ofstream res_file(name + ".dat", std::ios_base::out | std::ios_base::app);
    res_file.precision(17);
    res_file << std::fixed << accuracy << " " << time_running << std::endl;
}

int main(int nargs, char* args[])
{
    srand(time(NULL));
    bopt_params par = initialize_parameters_to_default();
    par.n_iterations = 190;
    par.n_inner_iterations = 250;
    par.n_iter_relearn = 0;
    //par.random_seed = 0;
    par.verbose_level = 0;
    par.noise = 1e-10;
    par.sigma_s = 1;
    par.sc_type = SC_ML;
    par.init_method = 3;
    strcpy(par.crit_name, "cLCB");
    par.crit_params[0] = 0.125;
    par.n_crit_params = 1;
    par.force_jump = 0;
    strcpy(par.kernel.name, "kMaternISO5");

    benchmark<BraninNormalized>(par, "branin");
    benchmark<Hartmann6>(par, "hartmann6");
    benchmark<Hartmann3>(par, "hartmann3");
    benchmark<Rastrigin>(par, "rastrigin");
    benchmark<Sphere>(par, "sphere");
    benchmark<Ellipsoid>(par, "ellipsoid");
    benchmark<GoldsteinPrice>(par, "goldsteinprice");
    benchmark<SixHumpCamel>(par, "sixhumpcamel");

    return 0;
}
