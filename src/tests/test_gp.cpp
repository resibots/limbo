#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE gp

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/kernel/matern_five_halfs.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_lf_opt.hpp>

using namespace limbo;

Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
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
};

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
    BOOST_CHECK(abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(abs((mu(0) - 5)) < 1);
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

    GP_t gp(1, 1);
    std::vector<Eigen::VectorXd> observations = {make_v1(5), make_v1(10), make_v1(5)};
    std::vector<Eigen::VectorXd> samples = {make_v1(1), make_v1(2), make_v1(3)};

    gp.compute(samples, observations, 0.0);

    Eigen::VectorXd mu;
    double sigma;
    std::tie(mu, sigma) = gp.query(make_v1(1));
    BOOST_CHECK(abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(2));
    BOOST_CHECK(abs((mu(0) - 10)) < 1);
    BOOST_CHECK(sigma < 1e-5);

    std::tie(mu, sigma) = gp.query(make_v1(3));
    BOOST_CHECK(abs((mu(0) - 5)) < 1);
    BOOST_CHECK(sigma < 1e-5);
}