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
#define BOOST_TEST_MODULE test_poegp
#define protected public

#include <boost/test/unit_test.hpp>

#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/model/poegp.hpp>
#include <limbo/model/poegp/spt_split.hpp>

// Check gradient via finite differences method
template <typename F>
std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> check_grad(const F& f, const Eigen::VectorXd& x, double e = 1e-4)
{
    Eigen::VectorXd analytic_result, finite_diff_result;

    limbo::opt::eval_t res = f(x, true);
    analytic_result = limbo::opt::grad(res);

    finite_diff_result = Eigen::VectorXd::Zero(x.size());
    for (int j = 0; j < x.size(); j++) {
        Eigen::VectorXd test1 = x, test2 = x;
        test1[j] -= e;
        test2[j] += e;
        double res1 = limbo::opt::fun(f(test1, false));
        double res2 = limbo::opt::fun(f(test2, false));
        finite_diff_result[j] = (res2 - res1) / (2.0 * e);
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

Eigen::VectorXd make_v1(double x)
{
    return limbo::tools::make_vector(x);
}

Eigen::VectorXd make_v2(double x1, double x2)
{
    Eigen::VectorXd v2(2);
    v2 << x1, x2;
    return v2;
}

struct Params {
    struct kernel : public limbo::defaults::kernel {
    };

    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
    };

    struct mean_constant : public limbo::defaults::mean_constant {
    };

    struct opt_rprop : public limbo::defaults::opt_rprop {
    };
};

BOOST_AUTO_TEST_CASE(test_gp_check_lf_grad)
{
    using namespace limbo;

    struct POEGPParams : public Params {
        struct model_poegp {
            BO_PARAM(int, expert_size, 20);
        };
    };

    using KF_t = kernel::SquaredExpARD<POEGPParams>;
    using Mean_t = mean::Constant<POEGPParams>;
    using GP_t = model::POEGP<POEGPParams, KF_t, Mean_t>;

    GP_t gp(4, 2);

    std::vector<Eigen::VectorXd> observations, samples, test_samples, test_samples_mean, test_samples_kernel_mean;
    double e = 1e-4;

    // Random samples and test samples
    int N = 40, M = 100;

    for (int i = 0; i < N; i++) {
        samples.push_back(tools::random_vector(4));
        Eigen::VectorXd ob(2);
        ob << std::cos(samples[i](0)), std::sin(samples[i](1));
        observations.push_back(ob);
    }

    for (int i = 0; i < M; i++) {
        test_samples.push_back(tools::random_vector(gp.kernel_function().h_params_size()));
    }

    gp.compute(samples, observations);

    model::gp::KernelLFOpt<Params>::KernelLFOptimization<GP_t> kernel_optimization(gp);

    Eigen::VectorXd results(M);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_optimization, test_samples[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);
}

BOOST_AUTO_TEST_CASE(test_gp_check_lf_grad_noise)
{
    using namespace limbo;

    struct POEGPParams : public Params {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct model_poegp {
            BO_PARAM(int, expert_size, 20);
        };
    };

    using KF_t = kernel::SquaredExpARD<POEGPParams>;
    using Mean_t = mean::Constant<POEGPParams>;
    using GP_t = model::POEGP<POEGPParams, KF_t, Mean_t>;

    GP_t gp(4, 2);

    std::vector<Eigen::VectorXd> observations, samples, test_samples, test_samples_mean, test_samples_kernel_mean;
    double e = 1e-4;

    // Random samples and test samples
    int N = 40, M = 100;

    for (int i = 0; i < N; i++) {
        samples.push_back(tools::random_vector(4));
        Eigen::VectorXd ob(2);
        ob << std::cos(samples[i](0)), std::sin(samples[i](1));
        observations.push_back(ob);
    }

    for (int i = 0; i < M; i++) {
        test_samples.push_back(tools::random_vector(gp.kernel_function().h_params_size()));
    }

    gp.compute(samples, observations);

    model::gp::KernelLFOpt<Params>::KernelLFOptimization<GP_t> kernel_optimization(gp);

    Eigen::VectorXd results(M);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_optimization, test_samples[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);
}

BOOST_AUTO_TEST_CASE(test_timing)
{
    using namespace limbo;
    constexpr size_t N = 10;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct POEGPParams : public Params {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct model_poegp {
            BO_PARAM(int, expert_size, 20);
        };
    };

    using KF_t = kernel::SquaredExpARD<POEGPParams>;
    using MF_t = mean::Constant<POEGPParams>;
    using GP_t = model::GP<POEGPParams, KF_t, MF_t, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;
    using POEGP_t = model::POEGP<POEGPParams, KF_t, MF_t, model::poegp::RandomSplit<POEGPParams>, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;

    for (size_t i = 0; i < N; i++) {
        std::vector<Eigen::VectorXd> observations;
        std::vector<Eigen::VectorXd> samples;
        tools::rgen_double_t rgen(0.0, 10);
        for (size_t i = 0; i < M; i++) {
            observations.push_back(make_v1(rgen.rand()));
            samples.push_back(make_v1(rgen.rand()));
        }

        GP_t gp;
        auto t1 = std::chrono::steady_clock::now();
        gp.compute(samples, observations, false);
        gp.optimize_hyperparams();
        auto time_full = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count();

        POEGP_t sgp;
        auto t2 = std::chrono::steady_clock::now();
        sgp.compute(samples, observations, false);
        sgp.optimize_hyperparams();
        auto time_sparse = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t2).count();

        if (time_full <= time_sparse)
            failures++;
    }

    BOOST_CHECK(double(failures) / double(N) < 0.1);
}

BOOST_AUTO_TEST_CASE(test_accuracy)
{
    using namespace limbo;
    constexpr size_t N = 10;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct POEGPParams : public Params {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct model_poegp {
            BO_PARAM(int, expert_size, 20);
        };
    };

    using KF_t = kernel::SquaredExpARD<POEGPParams>;
    using MF_t = mean::Constant<POEGPParams>;
    using GP_t = model::GP<POEGPParams, KF_t, MF_t, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;
    using POEGP_t = model::POEGP<POEGPParams, KF_t, MF_t, model::poegp::RandomSplit<POEGPParams>, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;

    for (size_t i = 0; i < N; i++) {

        std::vector<Eigen::VectorXd> observations;
        std::vector<Eigen::VectorXd> samples;
        tools::rgen_double_t rgen(-2., 2.);
        for (size_t i = 0; i < M; i++) {
            samples.push_back(make_v1(rgen.rand()));
            observations.push_back(make_v1(std::cos(samples.back()[0])));
        }

        std::vector<Eigen::VectorXd> test_observations;
        std::vector<Eigen::VectorXd> test_samples;
        for (size_t i = 0; i < M; i++) {
            test_samples.push_back(make_v1(rgen.rand()));
            test_observations.push_back(make_v1(std::cos(test_samples.back()[0])));
        }

        GP_t gp;
        gp.compute(samples, observations, false);
        gp.optimize_hyperparams();

        POEGP_t sgp;
        sgp.compute(samples, observations, false);
        sgp.optimize_hyperparams();

        bool failed = false;

        // check if normal GP and sparse GP produce very similar results
        // in the learned points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val;
            double gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_sigma - sgp_sigma) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar results
        // in the test points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val;
            double gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(test_samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(test_samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_sigma - sgp_sigma) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar errors
        // in the test points
        for (size_t i = 0; i < M; i++) {
            double gp_val = gp.mu(test_samples[i])[0];
            double sgp_val = sgp.mu(test_samples[i])[0];

            double gp_error_val = std::abs(gp_val - test_observations[i][0]);
            double sgp_error_val = std::abs(sgp_val - test_observations[i][0]);

            if (std::abs(gp_error_val - sgp_error_val) > 1e-2)
                failed = true;
        }

        if (failed)
            failures++;
    }

    BOOST_CHECK(double(failures) / double(N) < 0.1);
}

BOOST_AUTO_TEST_CASE(test_accuracy_spt)
{
    using namespace limbo;
    constexpr size_t N = 10;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct POEGPParams : public Params {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct model_poegp {
            BO_PARAM(int, expert_size, 20);
        };

        struct model_poegp_spt_split {
            BO_PARAM(double, tau, 0.);
        };
    };

    using KF_t = kernel::SquaredExpARD<POEGPParams>;
    using MF_t = mean::Constant<POEGPParams>;
    using GP_t = model::GP<POEGPParams, KF_t, MF_t, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;
    using POEGP_t = model::POEGP<POEGPParams, KF_t, MF_t, model::poegp::SPTSplit<POEGPParams>, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;

    for (size_t i = 0; i < N; i++) {

        std::vector<Eigen::VectorXd> observations;
        std::vector<Eigen::VectorXd> samples;
        tools::rgen_double_t rgen(-2., 2.);
        for (size_t i = 0; i < M; i++) {
            samples.push_back(make_v1(rgen.rand()));
            observations.push_back(make_v1(std::cos(samples.back()[0])));
        }

        std::vector<Eigen::VectorXd> test_observations;
        std::vector<Eigen::VectorXd> test_samples;
        for (size_t i = 0; i < M; i++) {
            test_samples.push_back(make_v1(rgen.rand()));
            test_observations.push_back(make_v1(std::cos(test_samples.back()[0])));
        }

        GP_t gp;
        gp.compute(samples, observations, false);
        gp.optimize_hyperparams();

        POEGP_t sgp;
        sgp.compute(samples, observations, false);
        sgp.optimize_hyperparams();

        bool failed = false;

        // check if normal GP and sparse GP produce very similar results
        // in the learned points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val;
            double gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_sigma - sgp_sigma) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar results
        // in the test points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val;
            double gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(test_samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(test_samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_sigma - sgp_sigma) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar errors
        // in the test points
        for (size_t i = 0; i < M; i++) {
            double gp_val = gp.mu(test_samples[i])[0];
            double sgp_val = sgp.mu(test_samples[i])[0];

            double gp_error_val = std::abs(gp_val - test_observations[i][0]);
            double sgp_error_val = std::abs(sgp_val - test_observations[i][0]);

            if (std::abs(gp_error_val - sgp_error_val) > 1e-2)
                failed = true;
        }

        if (failed)
            failures++;
    }

    BOOST_CHECK(double(failures) / double(N) < 0.1);
}

BOOST_AUTO_TEST_CASE(test_multi)
{
    using namespace limbo;
    constexpr size_t N = 10;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct POEGPParams : public Params {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct model_poegp {
            BO_PARAM(int, expert_size, 20);
        };
    };

    using KF_t = kernel::SquaredExpARD<POEGPParams>;
    using MF_t = mean::Constant<POEGPParams>;
    using GP_t = model::GP<POEGPParams, KF_t, MF_t, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;
    using POEGP_t = model::POEGP<POEGPParams, KF_t, MF_t, model::poegp::RandomSplit<POEGPParams>, model::gp::KernelLFOpt<POEGPParams, opt::Rprop<POEGPParams>>>;

    for (size_t i = 0; i < N; i++) {

        std::vector<Eigen::VectorXd> observations;
        std::vector<Eigen::VectorXd> samples;
        tools::rgen_double_t rgen(-2., 2.);
        for (size_t i = 0; i < M; i++) {
            samples.push_back(make_v2(rgen.rand(), rgen.rand()));
            observations.push_back(make_v2(std::cos(samples.back()[0]), std::cos(samples.back()[1])));
        }

        std::vector<Eigen::VectorXd> test_observations;
        std::vector<Eigen::VectorXd> test_samples;
        for (size_t i = 0; i < M; i++) {
            test_samples.push_back(make_v2(rgen.rand(), rgen.rand()));
            test_observations.push_back(make_v2(std::cos(test_samples.back()[0]), std::cos(test_samples.back()[1])));
        }

        GP_t gp;
        gp.compute(samples, observations, false);
        gp.optimize_hyperparams();

        POEGP_t sgp;
        sgp.compute(samples, observations, false);
        sgp.optimize_hyperparams();

        bool failed = false;

        // check if normal GP and sparse GP produce very similar results
        // in the learned points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val;
            double gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_val[1] - sgp_val[1]) > 1e-2 || std::abs(gp_sigma - sgp_sigma) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar results
        // in the test points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val;
            double gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(test_samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(test_samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-1 || std::abs(gp_val[1] - sgp_val[1]) > 1e-1 || std::abs(gp_sigma - sgp_sigma) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar errors
        // in the test points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val = gp.mu(test_samples[i]);
            Eigen::VectorXd sgp_val = sgp.mu(test_samples[i]);

            double gp_error_val = (gp_val - test_observations[i]).norm();
            double sgp_error_val = (sgp_val - test_observations[i]).norm();

            if (std::abs(gp_error_val - sgp_error_val) > 1e-1)
                failed = true;
        }

        if (failed)
            failures++;
    }

    BOOST_CHECK(double(failures) / double(N) < 0.1);
}
