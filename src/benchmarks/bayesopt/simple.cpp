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
#define SKIP_TRICKS
#define BAYES_OPT
#include "../limbo/testfunctions.hpp"
#include <chrono>
#include <fstream>
#include <string>

template <typename Function>
void benchmark(const bopt_params& par, const std::string& name)
{
    auto t1 = std::chrono::steady_clock::now();
    Benchmark<Function> benchmark(par);
    vectord result(Function::dim_in());
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
