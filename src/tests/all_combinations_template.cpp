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
#define FUSION_MAX_VECTOR_SIZE 20
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {

    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.001);
    };

    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 1);
    };

    struct kernel_maternthreehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct mean_constant {
        BO_PARAM(double, constant, 0);
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct acqui_gpucb : public defaults::acqui_gpucb {
    };

    struct opt_gridsearch {
        BO_PARAM(int, bins, 20);
    };
#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#endif
#ifdef USE_NLOPT
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#endif
    struct opt_rprop : public defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };

    struct init_gridsampling {
        BO_PARAM(int, bins, 5);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 5);
    };

    struct init_randomsamplinggrid {
        BO_PARAM(int, samples, 5);
        BO_PARAM(int, bins, 5);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 20);
    };

    struct stop_maxpredictedvalue {
        BO_PARAM(double, ratio, 2);
    };

    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    };
};

struct MeanEval {
    MeanEval(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP&) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2.5 * x(0);
        res(1) = -4.5 * x(1);
        return res;
    }
};

struct Average {
    using result_type = double;
    double operator()(const Eigen::VectorXd& x) const
    {
        return (x(0) + x(1)) / 2;
    }
};

struct StateEval {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 2);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        res(0) = 3 * x(0) + 5;
        res(1) = -5 * x(1) + 2;
        return res;
    }
};

int main()
{
    // clang-format off
    @declarations
    @optimizer.optimize(StateEval());
    @optimizer.best_observation();
    @optimizer.best_sample();
    @optimizer.optimize(StateEval(), Average(), true);
    @optimizer.best_observation(Average());
    @optimizer.best_sample(Average());
    // clang-format on
}
