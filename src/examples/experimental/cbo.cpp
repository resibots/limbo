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
#include <limbo/limbo.hpp>

#include <limbo/experimental/acqui/eci.hpp>
#include <limbo/experimental/bayes_opt/cboptimizer.hpp>

using namespace limbo;

struct Params {
    struct bayes_opt_cboptimizer : public defaults::bayes_opt_cboptimizer {
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 10);
    };

    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.01);
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
    BO_PARAM(size_t, dim_in, 1);
    BO_PARAM(size_t, dim_out, 1);
    BO_PARAM(size_t, nb_constraints, 1);

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
    using func_t = cosine;

    using Stop_t = boost::fusion::vector<stop::MaxIterations<Params>>;
    using Stat_t = boost::fusion::vector<stat::Samples<Params>,
        stat::BestObservations<Params>,
        stat::AggregatedObservations<Params>>;
    using Mean_t = mean::Constant<Params>;
    using Kernel_t = kernel::Exp<Params>;
    using GP_t = model::GP<Params, Kernel_t, Mean_t>;
    using Constrained_GP_t = model::GP<Params, Kernel_t, Mean_t>;

    using Acqui_t = experimental::acqui::ECI<Params, GP_t, Constrained_GP_t>;
    using Init_t = init::RandomSampling<Params>;

    experimental::bayes_opt::CBOptimizer<Params,
        modelfun<GP_t>,
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
