#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nlopt_test

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>
#include <limbo/opt/cmaes.hpp>

using namespace limbo;

struct Params {
    struct opt_cmaes : public limbo::defaults::opt_cmaes {
    };
};

opt::eval_t fsphere(const Eigen::VectorXd& params, bool g)
{
    return opt::no_grad(-params(0) * params(0) - params(1) * params(1));
}

BOOST_AUTO_TEST_CASE(test_cmaes_unbounded)
{
    Eigen::VectorXd g = limbo::opt::Cmaes<Params>()(fsphere, Eigen::VectorXd::Zero(2), false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}

BOOST_AUTO_TEST_CASE(test_cmaes_bounded)
{
    Eigen::VectorXd g = limbo::opt::Cmaes<Params>()(fsphere, Eigen::VectorXd::Zero(2), true);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}
