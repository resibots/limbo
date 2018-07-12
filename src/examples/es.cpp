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
#include <limbo/opt.hpp>
#include <limbo/tools.hpp>

#include <limbo/opt/es.hpp>

// this short tutorial shows how to use the optimization api of limbo (opt::)
using namespace limbo;

struct ParamsES {
    struct opt_es : public defaults::opt_es {
        /// size of population
        BO_PARAM(int, population, 2);

        /// sigma_sq - exploration parameter
        BO_PARAM(double, sigma_sq, 0.01 * 0.01);

        /// antithetic - turn on/off antithetic sampling
        BO_PARAM(bool, antithetic, true);

        /// rank_fitness - use ranking as fitness instead of true fitness
        BO_PARAM(bool, rank_fitness, false);

        /// normalize_fitness - normalize fitness (i.e., zero-mean, unit-variance)
        BO_PARAM(bool, normalize_fitness, false);

        /// beta - gradient estimate multiplier
        BO_PARAM(double, beta, 2.);

        /// alpha - approximate gradient information, [0,1]
        /// if set to 1: only ES
        /// if set to 0: only gradient
        BO_PARAM(double, alpha, 0.5);

        /// k - number of previous approx. gradients
        /// for orthonomal basis
        BO_PARAM(int, k, 2);
    };

    struct opt_adam {
        /// number of max iterations
        BO_PARAM(int, iterations, 10000);

        /// alpha - learning rate
        BO_PARAM(double, alpha, 0.01);

        /// β1
        BO_PARAM(double, b1, 0.9);

        /// β2
        BO_PARAM(double, b2, 0.999);

        /// norm epsilon for stopping
        BO_PARAM(double, eps_stop, 0.0);
    };
};

namespace global {
    thread_local std::mt19937 gen(randutils::auto_seed_128{}.base());
    thread_local std::normal_distribution<double> gaussian(0., 1.);

    Eigen::VectorXd bias;
} // namespace global

opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad = false)
{
    double v = -(params.array() - 0.5).square().sum();
    if (!eval_grad)
        return opt::no_grad(v);

    // generate re-sampled noise
    Eigen::VectorXd noise(params.size());
    for (int i = 0; i < noise.size(); i++)
        noise(i) = global::gaussian(global::gen);
    noise.normalize();

    Eigen::VectorXd grad = (-2 * params).array() + 1.0;
    double g_norm = grad.norm();

    grad.array() += g_norm * (noise.array() + global::bias.array());
    return {v, grad};
}

int main(int argc, char** argv)
{
    // generate random bias
    global::bias = Eigen::VectorXd::Zero(1000);
    for (int i = 0; i < global::bias.size(); i++)
        global::bias(i) = global::gaussian(global::gen);
    global::bias.normalize();

    opt::Adam<ParamsES> adam;
    Eigen::VectorXd res_adam = adam(my_function, tools::random_vector(1000).array() * 2. - 1., false);
    std::cout << "Result with Adam:\t" //<< res_lbfgs.transpose()
              << " -> " << my_function(res_adam).first << std::endl;

    opt::ES<ParamsES, opt::Adam<ParamsES>> es;
    Eigen::VectorXd res_es = es(my_function, tools::random_vector(1000).array() * 2. - 1., false);
    std::cout << "Result with ES:\t" // << res_es.transpose()
              << " -> " << my_function(res_es).first << std::endl;

    return 0;
}
