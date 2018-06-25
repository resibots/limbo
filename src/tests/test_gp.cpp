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
#define BOOST_TEST_MODULE test_gp
#define protected public

#include <boost/test/unit_test.hpp>

#include <limbo/acqui/ucb.hpp>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/matern_five_halves.hpp>
#include <limbo/kernel/matern_three_halves.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/mean/function_ard.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/model/gp/kernel_loo_opt.hpp>
#include <limbo/model/gp/kernel_mean_lf_opt.hpp>
#include <limbo/model/gp/mean_lf_opt.hpp>
#include <limbo/model/multi_gp.hpp>
#include <limbo/model/multi_gp/parallel_lf_opt.hpp>
#include <limbo/model/sparsified_gp.hpp>
#include <limbo/opt/grid_search.hpp>
#include <limbo/tools/macros.hpp>

using namespace limbo;

// Check gradient via finite differences method
template <typename F>
std::tuple<double, Eigen::VectorXd, Eigen::VectorXd> check_grad(const F& f, const Eigen::VectorXd& x, double e = 1e-4)
{
    Eigen::VectorXd analytic_result, finite_diff_result;

    opt::eval_t res = f(x, true);
    analytic_result = opt::grad(res);

    finite_diff_result = Eigen::VectorXd::Zero(x.size());
    for (int j = 0; j < x.size(); j++) {
        Eigen::VectorXd test1 = x, test2 = x;
        test1[j] -= e;
        test2[j] += e;
        double res1 = opt::fun(f(test1, false));
        double res2 = opt::fun(f(test2, false));
        finite_diff_result[j] = (res2 - res1) / (2.0 * e);
    }

    return std::make_tuple((analytic_result - finite_diff_result).norm(), analytic_result, finite_diff_result);
}

Eigen::VectorXd make_v1(double x)
{
    return tools::make_vector(x);
}

Eigen::VectorXd make_v2(double x1, double x2)
{
    Eigen::VectorXd v2(2);
    v2 << x1, x2;
    return v2;
}

struct Params {
    struct kernel : public defaults::kernel {
    };

    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };

    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 1);
        BO_PARAM(double, l, 0.25);
    };

    struct mean_constant : public defaults::mean_constant {
    };

    struct opt_rprop : public defaults::opt_rprop {
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
};

BOOST_AUTO_TEST_CASE(test_gp_check_lf_grad)
{
    using namespace limbo;

    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::FunctionARD<Params, mean::Constant<Params>>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

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
        test_samples_mean.push_back(tools::random_vector(gp.mean_function().h_params_size()));
        test_samples_kernel_mean.push_back(tools::random_vector(gp.kernel_function().h_params_size() + gp.mean_function().h_params_size()));
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

    model::gp::KernelMeanLFOpt<Params>::KernelMeanLFOptimization<GP_t> kernel_mean_optimization(gp);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_mean_optimization, test_samples_kernel_mean[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);

    model::gp::MeanLFOpt<Params>::MeanLFOptimization<GP_t> mean_optimization(gp);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(mean_optimization, test_samples_mean[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);
}

BOOST_AUTO_TEST_CASE(test_gp_check_lf_grad_noise)
{
    using namespace limbo;
    struct Parameters {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
        };

        struct kernel_maternfivehalves {
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 0.25);
        };

        struct mean_constant : public defaults::mean_constant {
        };

        struct opt_rprop : public defaults::opt_rprop {
        };

        struct acqui_ucb : public defaults::acqui_ucb {
        };

        struct opt_gridsearch : public defaults::opt_gridsearch {
        };
    };

    using KF_t = kernel::SquaredExpARD<Parameters>;
    using Mean_t = mean::FunctionARD<Params, mean::Constant<Parameters>>;
    using GP_t = model::GP<Parameters, KF_t, Mean_t>;

    GP_t gp(4, 2);

    std::vector<Eigen::VectorXd> observations, samples, test_samples, test_samples_kernel_mean;
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
        test_samples_kernel_mean.push_back(tools::random_vector(gp.kernel_function().h_params_size() + gp.mean_function().h_params_size()));
    }

    gp.compute(samples, observations);

    model::gp::KernelLFOpt<Parameters>::KernelLFOptimization<GP_t> kernel_optimization(gp);

    Eigen::VectorXd results(M);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_optimization, test_samples[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);

    model::gp::KernelMeanLFOpt<Parameters>::KernelMeanLFOptimization<GP_t> kernel_mean_optimization(gp);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_mean_optimization, test_samples_kernel_mean[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);
}

BOOST_AUTO_TEST_CASE(test_gp_check_loo_grad)
{
    using namespace limbo;

    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    GP_t gp(4, 2);

    std::vector<Eigen::VectorXd> observations, samples, test_samples;
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

    model::gp::KernelLooOpt<Params>::KernelLooOptimization<GP_t> kernel_optimization(gp);

    Eigen::VectorXd results(M);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_optimization, test_samples[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);
}

BOOST_AUTO_TEST_CASE(test_gp_check_loo_grad_noise)
{
    using namespace limbo;
    struct Parameters {
        struct kernel : public defaults::kernel {
            BO_PARAM(bool, optimize_noise, true);
        };

        struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
        };

        struct kernel_maternfivehalves {
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 0.25);
        };

        struct mean_constant : public defaults::mean_constant {
        };

        struct opt_rprop : public defaults::opt_rprop {
        };

        struct acqui_ucb : public defaults::acqui_ucb {
        };

        struct opt_gridsearch : public defaults::opt_gridsearch {
        };
    };

    using KF_t = kernel::SquaredExpARD<Parameters>;
    using Mean_t = mean::Constant<Parameters>;
    using GP_t = model::GP<Parameters, KF_t, Mean_t>;

    GP_t gp(4, 2);

    std::vector<Eigen::VectorXd> observations, samples, test_samples;
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

    model::gp::KernelLooOpt<Parameters>::KernelLooOptimization<GP_t> kernel_optimization(gp);

    Eigen::VectorXd results(M);

    for (int i = 0; i < M; i++) {
        auto res = check_grad(kernel_optimization, test_samples[i], 1e-4);
        results(i) = std::get<0>(res);
        // std::cout << std::get<1>(res).transpose() << " vs " << std::get<2>(res).transpose() << " --> " << results(i) << std::endl;
    }

    BOOST_CHECK(results.array().sum() < M * e);
}

BOOST_AUTO_TEST_CASE(test_gp_check_inv_kernel_computation)
{
    using namespace limbo;

    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::FunctionARD<Params, mean::Constant<Params>>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    GP_t gp(4, 2);

    std::vector<Eigen::VectorXd> observations, samples;

    // Random samples
    int N = 10;

    for (int i = 0; i < N; i++) {
        samples.push_back(tools::random_vector(4));
        Eigen::VectorXd ob(2);
        ob << std::cos(samples[i](0)), std::sin(samples[i](1));
        observations.push_back(ob);
    }

    gp.compute(samples, observations);

    // Check that variable is properly initialized
    BOOST_CHECK(!gp.inv_kernel_computed());

    // Check that kernel is computed
    gp.compute_inv_kernel();
    BOOST_CHECK(gp.inv_kernel_computed());

    // Check recompute alternatives
    gp.recompute(false, false);
    BOOST_CHECK(gp.inv_kernel_computed());

    gp.recompute(true, false);
    BOOST_CHECK(gp.inv_kernel_computed());

    gp.recompute(true, true);
    BOOST_CHECK(!gp.inv_kernel_computed());

    // Check add_sample
    gp.compute_inv_kernel();
    gp.add_sample(samples.back(), observations.back());
    BOOST_CHECK(!gp.inv_kernel_computed());

    // Check different implicit computations of inverse kernel
    gp.compute_kernel_grad_log_lik();
    BOOST_CHECK(gp.inv_kernel_computed());

    gp.recompute();
    BOOST_CHECK(!gp.inv_kernel_computed());
    gp.compute_mean_grad_log_lik();
    BOOST_CHECK(gp.inv_kernel_computed());

    gp.recompute();
    BOOST_CHECK(!gp.inv_kernel_computed());
    gp.compute_log_loo_cv();
    BOOST_CHECK(gp.inv_kernel_computed());

    gp.recompute();
    BOOST_CHECK(!gp.inv_kernel_computed());
    gp.compute_kernel_grad_log_loo_cv();
    BOOST_CHECK(gp.inv_kernel_computed());
}

BOOST_AUTO_TEST_CASE(test_gp_dim)
{
    using namespace limbo;

    using KF_t = kernel::MaternFiveHalves<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    GP_t gp; // no init with dim

    std::vector<Eigen::VectorXd> observations = {make_v2(5, 5), make_v2(10, 10),
        make_v2(5, 5)};
    std::vector<Eigen::VectorXd> samples = {make_v2(1, 1), make_v2(2, 2), make_v2(3, 3)};

    gp.compute(samples, observations);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v2(1, 1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(std::abs((mu(1) - 5)) < 1);

    BOOST_CHECK(sigma <= 2. * (Params::kernel::noise() + 1e-8));
}

BOOST_AUTO_TEST_CASE(test_gp)
{
    using namespace limbo;

    using KF_t = kernel::MaternFiveHalves<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    GP_t gp;
    std::vector<Eigen::VectorXd> observations = {make_v1(5), make_v1(10),
        make_v1(5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma <= 2. * (Params::kernel::noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(std::abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma <= 2. * (Params::kernel::noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma <= 2. * (Params::kernel::noise() + 1e-8));

    for (double x = 0; x < 4; x += 0.05) {
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp.query(make_v1(x));
        BOOST_CHECK(gp.mu(make_v1(x)) == mu);
        BOOST_CHECK(gp.sigma(make_v1(x)) == sigma);
        // std::cout << x << " " << mu << " " << mu.array() - sigma << " "
        //           << mu.array() + sigma << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(test_gp_identical_samples)
{
    using namespace limbo;

    using KF_t = kernel::MaternFiveHalves<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    GP_t gp;
    std::vector<Eigen::VectorXd> observations;
    std::vector<Eigen::VectorXd> samples;
    for (int i = 0; i < 10; i++) {
        samples.push_back(make_v1(1));
        observations.push_back(make_v1(std::cos(1)));
    }

    gp.compute(samples, observations);

    GP_t gp2;
    for (int i = 0; i < 10; i++) {
        gp2.add_sample(make_v1(1), make_v1(std::cos(1)));
    }

    // Compute kernel matrix
    Eigen::MatrixXd kernel;
    size_t n = samples.size();
    kernel.resize(n, n);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j <= i; ++j)
            kernel(i, j) = gp.kernel_function()(samples[i], samples[j], i, j);

    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < i; ++j)
            kernel(j, i) = kernel(i, j);

    // Reconstruct kernels from cholesky decomposition
    Eigen::MatrixXd reconstructed_kernel = gp.matrixL() * gp.matrixL().transpose();
    Eigen::MatrixXd reconstructed_kernel2 = gp2.matrixL() * gp2.matrixL().transpose();

    // Check if the reconstructed kernels match the actual one
    BOOST_CHECK(kernel.isApprox(reconstructed_kernel, 1e-5));
    BOOST_CHECK(kernel.isApprox(reconstructed_kernel2, 1e-5));

    // Check if incremental cholesky produces the same result
    Eigen::VectorXd mu1, mu2;
    double s1, s2;

    std::tie(mu1, s1) = gp.query(make_v1(1));
    std::tie(mu2, s2) = gp2.query(make_v1(1));

    BOOST_CHECK((mu1 - mu2).norm() < 1e-4);
    BOOST_CHECK(std::sqrt((s1 - s2) * (s1 - s2)) < 1e-4);
}

BOOST_AUTO_TEST_CASE(test_gp_bw_inversion)
{
    using namespace limbo;
    size_t N = 1000;
    size_t failures = 0;

    using KF_t = kernel::MaternFiveHalves<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    for (size_t i = 0; i < N; i++) {

        std::vector<Eigen::VectorXd> observations;
        std::vector<Eigen::VectorXd> samples;
        tools::rgen_double_t rgen(0.0, 10);
        for (size_t i = 0; i < 100; i++) {
            observations.push_back(make_v1(rgen.rand()));
            samples.push_back(make_v1(rgen.rand()));
        }

        GP_t gp;
        auto t1 = std::chrono::steady_clock::now();
        gp.compute(samples, observations);
        // auto time_init = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count();
        // std::cout.precision(17);
        // std::cout << "Time running first batch: " << time_init << "us" << std::endl
        //           << std::endl;

        observations.push_back(make_v1(rgen.rand()));
        samples.push_back(make_v1(rgen.rand()));

        t1 = std::chrono::steady_clock::now();
        gp.add_sample(samples.back(), observations.back());
        auto time_increment = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count();
        // std::cout << "Time running increment: " << time_increment << "us" << std::endl
        //           << std::endl;

        t1 = std::chrono::steady_clock::now();
        gp.recompute(true);
        auto time_recompute = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count();
        // std::cout << "Time recomputing: " << time_recompute << "us" << std::endl
        //           << std::endl;

        GP_t gp2;
        t1 = std::chrono::steady_clock::now();
        gp2.compute(samples, observations);
        auto time_full = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t1).count();
        // std::cout << "Time running whole batch: " << time_full << "us" << std::endl
        //           << std::endl;

        bool failed = false;

        Eigen::VectorXd s = make_v1(rgen.rand());
        if ((gp.mu(s) - gp2.mu(s)).norm() >= 1e-5)
            failed = true;
        if (!gp.matrixL().isApprox(gp2.matrixL(), 1e-5))
            failed = true;
        if (time_full <= time_increment)
            failed = true;
        if (time_recompute <= time_increment)
            failed = true;

        if (failed)
            failures++;
    }

    BOOST_CHECK(double(failures) / double(N) < 0.1);
}

BOOST_AUTO_TEST_CASE(test_gp_no_samples_acqui_opt)
{
    using namespace limbo;

    struct FirstElem {
        double operator()(const Eigen::VectorXd& x) const
        {
            return x(0);
        }
    };

    using acquiopt_t = opt::GridSearch<Params>;

    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;
    using acquisition_function_t = acqui::UCB<Params, GP_t>;

    GP_t gp(2, 2);

    acquisition_function_t acqui(gp, 0);
    acquiopt_t acqui_optimizer;

    // we do not have gradient in our current acquisition function
    auto acqui_optimization =
        [&](const Eigen::VectorXd& x, bool g) { return acqui(x, FirstElem(), g); };
    Eigen::VectorXd starting_point = tools::random_vector(2);
    Eigen::VectorXd test = acqui_optimizer(acqui_optimization, starting_point, true);
    BOOST_CHECK(test(0) < 1e-5);
    BOOST_CHECK(test(1) < 1e-5);
}

BOOST_AUTO_TEST_CASE(test_gp_auto)
{
    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t, model::gp::KernelLFOpt<Params>>;

    GP_t gp;
    std::vector<Eigen::VectorXd> observations = {make_v1(5), make_v1(10), make_v1(5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations, false);
    gp.optimize_hyperparams();

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma <= 2. * (gp.kernel_function().noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(std::abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma <= 2. * (gp.kernel_function().noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma <= 2. * (gp.kernel_function().noise() + 1e-8));
}

BOOST_AUTO_TEST_CASE(test_gp_init_variance)
{
    using namespace limbo;

    struct Parameters {
        struct kernel : public defaults::kernel {
        };

        struct kernel_squared_exp_ard {
            BO_PARAM(int, k, 0);
            BO_PARAM(double, sigma_sq, 10);
        };
        struct kernel_exp {
            BO_PARAM(double, sigma_sq, 10);
            BO_PARAM(double, l, 1);
        };
        struct kernel_maternthreehalves {
            BO_PARAM(double, sigma_sq, 10);
            BO_PARAM(double, l, 0.25);
        };
        struct kernel_maternfivehalves {
            BO_PARAM(double, sigma_sq, 10);
            BO_PARAM(double, l, 0.25);
        };
    };

    // MaternThreeHalves
    using GP1_t = model::GP<Params, kernel::MaternThreeHalves<Parameters>, mean::Constant<Params>>;

    GP1_t gp1(1, 1);

    double sigma = gp1.sigma(tools::random_vector(1));

    BOOST_CHECK_CLOSE(sigma, 10.0, 1);

    // MaternFiveHalves
    using GP2_t = model::GP<Params, kernel::MaternFiveHalves<Parameters>, mean::Constant<Params>>;

    GP2_t gp2(1, 1);

    sigma = gp2.sigma(tools::random_vector(1));

    BOOST_CHECK_CLOSE(sigma, 10.0, 1);

    // Exponential
    using GP3_t = model::GP<Params, kernel::Exp<Parameters>, mean::Constant<Params>>;

    GP3_t gp3(1, 1);

    sigma = gp3.sigma(tools::random_vector(1));

    BOOST_CHECK_CLOSE(sigma, 10.0, 1);

    // ARD Squared Exponential
    using GP4_t = model::GP<Params, kernel::SquaredExpARD<Parameters>, mean::Constant<Params>>;

    GP4_t gp4(1, 1);

    sigma = gp4.sigma(tools::random_vector(1));

    BOOST_CHECK_CLOSE(sigma, 10.0, 1);
}

BOOST_AUTO_TEST_CASE(test_sparse_gp)
{
    using namespace limbo;
    constexpr size_t N = 10;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct SparseParams {
        struct mean_constant : public defaults::mean_constant {
        };
        struct kernel : public defaults::kernel {
        };
        struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
        };
        struct opt_rprop : public defaults::opt_rprop {
        };
        struct model_sparse_gp {
            BO_PARAM(int, max_points, M / 3);
        };
    };

    using KF_t = kernel::SquaredExpARD<SparseParams>;
    using MF_t = mean::Constant<SparseParams>;
    using GP_t = model::GP<SparseParams, KF_t, MF_t, model::gp::KernelLFOpt<SparseParams, opt::Rprop<SparseParams>>>;
    using SparsifiedGP_t = model::SparsifiedGP<SparseParams, KF_t, MF_t, model::gp::KernelLFOpt<SparseParams, opt::Rprop<SparseParams>>>;

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

        SparsifiedGP_t sgp;
        auto t2 = std::chrono::steady_clock::now();
        sgp.compute(samples, observations, false);
        sgp.optimize_hyperparams();
        auto time_sparse = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t2).count();

        if (time_full <= time_sparse)
            failures++;
    }

    BOOST_CHECK(double(failures) / double(N) < 0.1);
}

BOOST_AUTO_TEST_CASE(test_sparse_gp_accuracy)
{
    using namespace limbo;
    constexpr size_t N = 20;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct SparseParams {
        struct mean_constant : public defaults::mean_constant {
        };
        struct kernel : public defaults::kernel {
        };
        struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
        };
        struct opt_rprop : public defaults::opt_rprop {
        };
        struct model_sparse_gp {
            BO_PARAM(int, max_points, M / 2);
        };
    };

    using KF_t = kernel::SquaredExpARD<SparseParams>;
    using MF_t = mean::Constant<SparseParams>;
    using GP_t = model::GP<SparseParams, KF_t, MF_t, model::gp::KernelLFOpt<SparseParams, opt::Rprop<SparseParams>>>;
    using SparsifiedGP_t = model::SparsifiedGP<SparseParams, KF_t, MF_t, model::gp::KernelLFOpt<SparseParams, opt::Rprop<SparseParams>>>;

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

        SparsifiedGP_t sgp;
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

BOOST_AUTO_TEST_CASE(test_multi_gp_dim)
{
    using namespace limbo;

    using KF_t = kernel::MaternFiveHalves<Params>;
    using Mean_t = mean::Constant<Params>;
    using GP_t = model::GP<Params, KF_t, Mean_t>;

    GP_t gp; // no init with dim

    std::vector<Eigen::VectorXd> observations = {make_v2(5, 5), make_v2(10, 10),
        make_v2(5, 5)};
    std::vector<Eigen::VectorXd> samples = {make_v2(1, 1), make_v2(2, 2), make_v2(3, 3)};

    gp.compute(samples, observations);

    using MultiGP_t = model::MultiGP<Params, model::GP, KF_t, Mean_t>;
    MultiGP_t multi_gp;
    multi_gp.compute(samples, observations);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v2(1, 1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(std::abs((mu(1) - 5)) < 1);

    BOOST_CHECK(sigma <= 2. * (Params::kernel::noise() + 1e-8));

    Eigen::VectorXd mu_multi, sigma_multi;

    std::tie(mu_multi, sigma_multi) = multi_gp.query(make_v2(1, 1));
    BOOST_CHECK(std::abs((mu_multi(0) - 5)) < 1);
    BOOST_CHECK(std::abs((mu_multi(1) - 5)) < 1);

    BOOST_CHECK(sigma_multi(0) <= 2. * (Params::kernel::noise() + 1e-8));
    BOOST_CHECK(sigma_multi(1) <= 2. * (Params::kernel::noise() + 1e-8));

    BOOST_CHECK(std::abs(mu_multi(0) - mu(0)) < 1e-6);
    BOOST_CHECK(std::abs(mu_multi(1) - mu(1)) < 1e-6);
    BOOST_CHECK(std::abs(sigma_multi(0) - sigma) < 1e-6);
    BOOST_CHECK(std::abs(sigma_multi(1) - sigma) < 1e-6);
}

BOOST_AUTO_TEST_CASE(test_multi_gp)
{
    using namespace limbo;

    using KF_t = kernel::MaternFiveHalves<Params>;
    using Mean_t = mean::Constant<Params>;
    using MultiGP_t = model::MultiGP<Params, model::GP, KF_t, Mean_t>;

    MultiGP_t gp;
    std::vector<Eigen::VectorXd> observations = {make_v1(5), make_v1(10),
        make_v1(5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations);

    Eigen::VectorXd mu, sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma(0) <= 2. * (Params::kernel::noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(std::abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma(0) <= 2. * (Params::kernel::noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma(0) <= 2. * (Params::kernel::noise() + 1e-8));

    for (double x = 0; x < 4; x += 0.05) {
        Eigen::VectorXd mu, sigma;
        std::tie(mu, sigma) = gp.query(make_v1(x));
        BOOST_CHECK(gp.mu(make_v1(x)) == mu);
        BOOST_CHECK(gp.sigma(make_v1(x)) == sigma);
    }
}

BOOST_AUTO_TEST_CASE(test_sparse_multi_gp)
{
    using namespace limbo;
    constexpr size_t N = 20;
    constexpr size_t M = 100;
    size_t failures = 0;

    struct SparseParams {
        struct mean_constant : public defaults::mean_constant {
        };
        struct kernel : public defaults::kernel {
        };
        struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
        };
        struct opt_rprop : public defaults::opt_rprop {
        };
        struct model_sparse_gp {
            BO_PARAM(int, max_points, M / 2);
        };
    };

    using KF_t = kernel::SquaredExpARD<SparseParams>;
    using MF_t = mean::Constant<SparseParams>;
    using MultiGP_t = model::MultiGP<SparseParams, model::GP, KF_t, MF_t, model::multi_gp::ParallelLFOpt<SparseParams, model::gp::KernelLFOpt<SparseParams>>>;
    using MultiSparseGP_t = model::MultiGP<SparseParams, model::SparsifiedGP, KF_t, MF_t, model::multi_gp::ParallelLFOpt<SparseParams, model::gp::KernelLFOpt<SparseParams>>>;

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

        MultiGP_t gp;
        gp.compute(samples, observations, false);
        gp.optimize_hyperparams();

        MultiSparseGP_t sgp;
        sgp.compute(samples, observations, false);
        sgp.optimize_hyperparams();

        bool failed = false;

        // check if normal GP and sparse GP produce very similar results
        // in the learned points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val, gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_sigma[0] - sgp_sigma[0]) > 1e-2)
                failed = true;
        }

        // check if normal GP and sparse GP produce very similar results
        // in the test points
        for (size_t i = 0; i < M; i++) {
            Eigen::VectorXd gp_val, sgp_val, gp_sigma, sgp_sigma;
            std::tie(gp_val, gp_sigma) = gp.query(test_samples[i]);
            std::tie(sgp_val, sgp_sigma) = sgp.query(test_samples[i]);

            if (std::abs(gp_val[0] - sgp_val[0]) > 1e-2 || std::abs(gp_sigma[0] - sgp_sigma[0]) > 1e-2)
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

BOOST_AUTO_TEST_CASE(test_multi_gp_auto)
{
    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::Constant<Params>;
    using MultiGP_t = model::MultiGP<Params, model::GP, KF_t, Mean_t, model::multi_gp::ParallelLFOpt<Params, model::gp::KernelLFOpt<Params>>>;

    MultiGP_t gp;
    std::vector<Eigen::VectorXd> observations = {make_v2(5, 5), make_v2(10, 10), make_v2(5, 5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations, false);
    gp.optimize_hyperparams();

    Eigen::VectorXd mu, sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma(0) <= 2. * (gp.gp_models()[0].kernel_function().noise() + 1e-8));
    BOOST_CHECK(sigma(1) <= 2. * (gp.gp_models()[1].kernel_function().noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(std::abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma(0) <= 2. * (gp.gp_models()[0].kernel_function().noise() + 1e-8));
    BOOST_CHECK(sigma(1) <= 2. * (gp.gp_models()[1].kernel_function().noise() + 1e-8));

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma(0) <= 2. * (gp.gp_models()[0].kernel_function().noise() + 1e-8));
    BOOST_CHECK(sigma(1) <= 2. * (gp.gp_models()[1].kernel_function().noise() + 1e-8));
}

BOOST_AUTO_TEST_CASE(test_multi_gp_recompute)
{
    using KF_t = kernel::SquaredExpARD<Params>;
    using Mean_t = mean::Constant<Params>;
    using MultiGP_t = model::MultiGP<Params, model::GP, KF_t, Mean_t>;

    MultiGP_t gp;

    gp.add_sample(make_v2(1, 1), make_v1(2));
    gp.add_sample(make_v2(2, 2), make_v1(10));

    // make sure that the observations are properly passed
    BOOST_CHECK(gp._observations[0](0) == 2.);
    BOOST_CHECK(gp._observations[1](0) == 10.);

    // make sure the child GPs have the proper observations
    BOOST_CHECK(gp.gp_models()[0]._observations.row(0)[0] == (2. - gp.mean_function().h_params()[0]));
    BOOST_CHECK(gp.gp_models()[0]._observations.row(1)[0] == (10. - gp.mean_function().h_params()[0]));

    // now change the mean function parameters
    gp.mean_function().set_h_params(make_v1(2));

    // make sure that the observations are properly passed
    BOOST_CHECK(gp._observations[0](0) == 2.);
    BOOST_CHECK(gp._observations[1](0) == 10.);

    // make sure the child GPs do not have the proper observations
    BOOST_CHECK(!(gp.gp_models()[0]._observations.row(0)[0] == (2. - gp.mean_function().h_params()[0])));
    BOOST_CHECK(!(gp.gp_models()[0]._observations.row(1)[0] == (10. - gp.mean_function().h_params()[0])));

    // recompute the GP
    gp.recompute();

    // make sure that the observations are properly passed
    BOOST_CHECK(gp._observations[0](0) == 2.);
    BOOST_CHECK(gp._observations[1](0) == 10.);

    // make sure the child GPs have the proper observations
    BOOST_CHECK(gp.gp_models()[0]._observations.row(0)[0] == (2. - gp.mean_function().h_params()[0]));
    BOOST_CHECK(gp.gp_models()[0]._observations.row(1)[0] == (10. - gp.mean_function().h_params()[0]));
}
