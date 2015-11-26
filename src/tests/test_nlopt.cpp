#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nlopt_test

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>
#include <limbo/opt/nlopt_grad.hpp>
#include <limbo/opt/nlopt_no_grad.hpp>

struct Params {
    struct nlopt {
        BO_PARAM(int, iters, 80);
    };
};

struct TestOpt {
public:
    TestOpt() {}

    double utility(const Eigen::VectorXd& params) const
    {
        return -params(0) * params(0) - params(1) * params(1);
    }

    std::pair<double, Eigen::VectorXd> utility_and_grad(const Eigen::VectorXd& params) const
    {
        double v = -params(0) * params(0) - params(1) * params(1);
        Eigen::VectorXd grad(2);
        grad(0) = -2 * params(0);
        grad(1) = -2 * params(1);
        return std::make_pair(v, grad);
    }

    size_t param_size() const
    {
        return 2;
    }

    Eigen::VectorXd init() const
    {
        return (Eigen::VectorXd::Random(param_size()).array() + 1) / 2.0;
    }
};

BOOST_AUTO_TEST_CASE(test_nlopt_grad_simple)
{
    TestOpt util;
    Eigen::VectorXd g = limbo::opt::NLOptGrad<Params, nlopt::LD_MMA>()(util, false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}

BOOST_AUTO_TEST_CASE(test_nlopt_no_grad_simple)
{
    TestOpt util;
    Eigen::VectorXd g = limbo::opt::NLOptNoGrad<Params, nlopt::LN_COBYLA>()(util, false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}
