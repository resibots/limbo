#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE gp

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/kernel/matern_five_halfs.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>
#include <limbo/acqui/ucb.hpp>
#include <limbo/opt/grid_search.hpp>

using namespace limbo;

Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
}

Eigen::VectorXd make_v2(double x1, double x2)
{
    Eigen::VectorXd v2(2);
    v2 << x1, x2;
    return v2;
}

struct Params {
    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.25);
    };

    struct mean_constant : public defaults::mean_constant {
    };

    struct opt_rprop : public defaults::opt_rprop {
    };

    struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
};

BOOST_AUTO_TEST_CASE(test_gp_dim)
{
    using namespace limbo;

    typedef kernel::MaternFiveHalfs<Params> KF_t;
    typedef mean::Constant<Params> Mean_t;
    typedef model::GP<Params, KF_t, Mean_t> GP_t;

    GP_t gp; // no init with dim

    std::vector<Eigen::VectorXd> observations = {make_v2(5, 5), make_v2(10, 10),
        make_v2(5, 5)};
    std::vector<Eigen::VectorXd> samples = {make_v2(1, 1), make_v2(2, 2), make_v2(3, 3)};

    gp.compute(samples, observations, 0.0);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v2(1, 1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(std::abs((mu(1) - 5)) < 1);

    BOOST_CHECK(sigma < 1e-5);
}

BOOST_AUTO_TEST_CASE(test_gp)
{
    using namespace limbo;

    typedef kernel::MaternFiveHalfs<Params> KF_t;
    typedef mean::Constant<Params> Mean_t;
    typedef model::GP<Params, KF_t, Mean_t> GP_t;

    GP_t gp;
    std::vector<Eigen::VectorXd> observations = {make_v1(5), make_v1(10),
        make_v1(5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations, 0.0);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(std::abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    for (double x = 0; x < 4; x += 0.05) {
        Eigen::VectorXd mu;
        double sigma;
        std::tie(mu, sigma) = gp.query(make_v1(x));
        BOOST_CHECK(gp.mu(make_v1(x)) == mu);
        BOOST_CHECK(gp.sigma(make_v1(x)) == sigma);
        std::cout << x << " " << mu << " " << mu.array() - sigma << " "
                  << mu.array() + sigma << std::endl;
    }
}

BOOST_AUTO_TEST_CASE(test_gp_no_samples_acqui_opt)
{
    using namespace limbo;

    struct FirstElem {
        typedef double result_type;
        double operator()(const Eigen::VectorXd& x) const
        {
            return x(0);
        }
    };

    typedef opt::GridSearch<Params> acquiopt_t;

    typedef kernel::SquaredExpARD<Params> KF_t;
    typedef mean::Constant<Params> Mean_t;
    typedef model::GP<Params, KF_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> acquisition_function_t;

    GP_t gp(2, 2);

    acquisition_function_t acqui(gp, 0);
    acquiopt_t acqui_optimizer;

    // we do not have gradient in our current acquisition function
    auto acqui_optimization =
        [&](const Eigen::VectorXd& x, bool g) { return opt::no_grad(acqui(x, FirstElem())); };
    Eigen::VectorXd starting_point = tools::random_vector(2);
    Eigen::VectorXd test = acqui_optimizer(acqui_optimization, starting_point, true);
    BOOST_CHECK(test(0) < 1e-5);
    BOOST_CHECK(test(1) < 1e-5);
}

BOOST_AUTO_TEST_CASE(test_gp_blacklist)
{
    using namespace limbo;

    typedef kernel::MaternFiveHalfs<Params> KF_t;
    typedef mean::Constant<Params> Mean_t;
    typedef model::GP<Params, KF_t, Mean_t> GP_t;

    GP_t gp;
    std::vector<Eigen::VectorXd> samples = {make_v1(1)};
    std::vector<Eigen::VectorXd> observations = {make_v1(5)};
    std::vector<Eigen::VectorXd> bl_samples = {make_v1(2)};

    gp.compute(samples, observations, 0.0);

    Eigen::VectorXd prev_mu1, mu1, prev_mu2, mu2;
    double prev_sigma1, sigma1, prev_sigma2, sigma2;

    std::tie(prev_mu1, prev_sigma1) = gp.query(make_v1(1));
    std::tie(prev_mu2, prev_sigma2) = gp.query(make_v1(2));

    gp.compute(samples, observations, 0.0, bl_samples);

    std::tie(mu1, sigma1) = gp.query(make_v1(1));
    std::tie(mu2, sigma2) = gp.query(make_v1(2));

    BOOST_CHECK(prev_mu1 == mu1);
    BOOST_CHECK(prev_sigma1 == sigma1);
    BOOST_CHECK(prev_mu2 == mu2);
    BOOST_CHECK(prev_sigma2 > sigma2);
    BOOST_CHECK(sigma2 == 0);
}

BOOST_AUTO_TEST_CASE(test_gp_auto)
{
    typedef kernel::SquaredExpARD<Params> KF_t;
    typedef mean::Constant<Params> Mean_t;
    typedef model::GP<Params, KF_t, Mean_t, model::gp::KernelLFOpt<Params>> GP_t;

    GP_t gp;
    std::vector<Eigen::VectorXd> observations = {make_v1(5), make_v1(10), make_v1(5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations, 0.0);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(std::abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(std::abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);
}
