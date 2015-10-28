
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE parallel

#include <boost/test/unit_test.hpp>

#include <limbo/misc/macros.hpp>
#include <limbo/models/gp.hpp>

using namespace limbo;

Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
}

struct Params {
    struct kf_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.25);
    };

    struct meanconstant {
        static Eigen::VectorXd constant() { return make_v1(0.0); };
    };
};

BOOST_AUTO_TEST_CASE(test_gp)
{
    using namespace limbo;

    typedef kernel_fun::MaternFiveHalfs<Params> KF_t;
    typedef mean_fun::Constant<Params> Mean_t;
    typedef models::GP<Params, KF_t, Mean_t> GP_t;

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

BOOST_AUTO_TEST_CASE(test_blacklist)
{
    using namespace limbo;

    typedef kernel_fun::MaternFiveHalfs<Params> KF_t;
    typedef mean_fun::Constant<Params> Mean_t;
    typedef models::GP<Params, KF_t, Mean_t> GP_t;

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
