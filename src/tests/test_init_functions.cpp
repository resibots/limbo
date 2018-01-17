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
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_init_functions

#include <boost/test/unit_test.hpp>

#include <limbo/acqui.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/init.hpp>
#include <limbo/tools/macros.hpp>

using namespace limbo;

struct Params {
    struct bayes_opt_bobase : public defaults::bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, false);
    };

    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 0);
    };

    struct kernel : public defaults::kernel {
        BO_PARAM(double, noise, 0.01);
    };

    struct kernel_maternfivehalves : public defaults::kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 0.25);
    };

    struct acqui_ucb : public defaults::acqui_ucb {
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

struct fit_eval {
    BO_PARAM(size_t, dim_in, 2);
    BO_PARAM(size_t, dim_out, 1);

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        double res = 0;
        for (int i = 0; i < x.size(); i++)
            res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
        return tools::make_vector(res);
    }
};

BOOST_AUTO_TEST_CASE(no_init)
{
    std::cout << "NoInit" << std::endl;
    using Init_t = init::NoInit<Params>;
    using Opt_t = bayes_opt::BOptimizer<Params, initfun<Init_t>>;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 0);
    BOOST_CHECK(opt.samples().size() == 0);
}

BOOST_AUTO_TEST_CASE(random_lhs)
{
    std::cout << "LHS" << std::endl;
    struct MyParams : public Params {
        struct init_lhs {
            BO_PARAM(int, samples, 10);
        };
    };

    using Init_t = init::LHS<MyParams>;
    using Opt_t = bayes_opt::BOptimizer<MyParams, initfun<Init_t>>;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 10);
    BOOST_CHECK(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
            BOOST_CHECK(i == 0 || x[i] != x[0]);
        }
    }
}

BOOST_AUTO_TEST_CASE(random_sampling)
{
    std::cout << "RandomSampling" << std::endl;
    struct MyParams : public Params {
        struct init_randomsampling {
            BO_PARAM(int, samples, 10);
        };
    };

    using Init_t = init::RandomSampling<MyParams>;
    using Opt_t = bayes_opt::BOptimizer<MyParams, initfun<Init_t>>;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 10);
    BOOST_CHECK(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
            BOOST_CHECK(i == 0 || x[i] != x[0]);
        }
    }
}

BOOST_AUTO_TEST_CASE(random_sampling_grid)
{
    std::cout << "RandomSamplingGrid" << std::endl;
    struct MyParams : public Params {
        struct init_randomsamplinggrid {
            BO_PARAM(int, samples, 10);
            BO_PARAM(int, bins, 4);
        };
    };

    using Init_t = init::RandomSamplingGrid<MyParams>;
    using Opt_t = bayes_opt::BOptimizer<MyParams, initfun<Init_t>>;

    Opt_t opt;
    opt.optimize(fit_eval());
    BOOST_CHECK(opt.observations().size() == 10);
    BOOST_CHECK(opt.samples().size() == 10);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
            BOOST_CHECK(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
        }
    }
}

BOOST_AUTO_TEST_CASE(grid_sampling)
{
    std::cout << "GridSampling" << std::endl;
    struct MyParams : public Params {
        struct init_gridsampling {
            BO_PARAM(int, bins, 4);
        };
    };

    using Init_t = init::GridSampling<MyParams>;
    using Opt_t = bayes_opt::BOptimizer<MyParams, initfun<Init_t>>;

    Opt_t opt;
    opt.optimize(fit_eval());
    std::cout << opt.observations().size() << std::endl;
    BOOST_CHECK(opt.observations().size() == 25);
    BOOST_CHECK(opt.samples().size() == 25);
    for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
        const Eigen::VectorXd& x = opt.samples()[j];
        std::cout << x.transpose() << std::endl;
        for (int i = 0; i < x.size(); ++i) {
            BOOST_CHECK(x[i] >= 0);
            BOOST_CHECK(x[i] <= 1);
            BOOST_CHECK(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
        }
    }
}
