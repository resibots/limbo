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

// this short tutorial shows how to use the optimization api of limbo (opt::)
using namespace limbo;

#ifdef USE_NLOPT
struct ParamsGrad {
    struct opt_nloptgrad : public defaults::opt_nloptgrad {
        BO_PARAM(int, iterations, 80);
    };
};

struct ParamsNoGrad {
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
        BO_PARAM(int, iterations, 80);
    };
};
#endif

#ifdef USE_LIBCMAES
struct ParamsCMAES {
    struct opt_cmaes : public defaults::opt_cmaes {
    };
};
#endif

// we maximize -(x_1-0.5)^2 - (x_2-0.5)^2
// the maximum is [0.5, 0.5] (f([0.5, 0.5] = 0))
opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad = false)
{
    double v = -(params.array() - 0.5).square().sum();
    if (!eval_grad)
        return opt::no_grad(v);
    Eigen::VectorXd grad = (-2 * params).array() + 1.0;
    return {v, grad};
}

int main(int argc, char** argv)
{
#ifdef USE_NLOPT
    // the type of the optimizer (here NLOpt with the LN_LBGFGS algorithm)
    opt::NLOptGrad<ParamsGrad, nlopt::LD_LBFGS> lbfgs;
    // we start from a random point (in 2D), and the search is not bounded
    Eigen::VectorXd res_lbfgs = lbfgs(my_function, tools::random_vector(2), false);
    std::cout << "Result with LBFGS:\t" << res_lbfgs.transpose()
              << " -> " << my_function(res_lbfgs).first << std::endl;

    // we can also use a gradient-free algorith, like DIRECT
    opt::NLOptNoGrad<ParamsNoGrad, nlopt::GN_DIRECT> direct;
    // we start from a random point (in 2D), and the search is bounded in [0,1]
    // be careful that DIRECT does not support unbounded search
    Eigen::VectorXd res_direct = direct(my_function, tools::random_vector(2), true);
    std::cout << "Result with DIRECT:\t" << res_direct.transpose()
              << " -> " << my_function(res_direct).first << std::endl;

#endif

#ifdef USE_LIBCMAES
    // or Cmaes
    opt::Cmaes<ParamsCMAES> cmaes;
    Eigen::VectorXd res_cmaes = cmaes(my_function, tools::random_vector(2), false);
    std::cout << "Result with CMA-ES:\t" << res_cmaes.transpose()
              << " -> " << my_function(res_cmaes).first << std::endl;

#endif

    return 0;
}
