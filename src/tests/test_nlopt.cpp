#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nlopt_test

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>
#include <limbo/opt/nlopt_grad.hpp>
#include <limbo/opt/nlopt_no_grad.hpp>

using namespace limbo;

struct Params {
    struct opt_nloptgrad {
        BO_PARAM(int, iterations, 80);
    };

    struct opt_nloptnograd {
        BO_PARAM(int, iterations, 80);
    };
};

opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad)
{
    double v = -params(0) * params(0) - params(1) * params(1);
    if (!eval_grad)
        return opt::no_grad(v);
    Eigen::VectorXd grad(2);
    grad(0) = -2 * params(0);
    grad(1) = -2 * params(1);
    return {v, grad};
}

BOOST_AUTO_TEST_CASE(test_nlopt_grad_simple)
{
    opt::NLOptGrad<Params, nlopt::LD_MMA> optimizer;
    Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}

BOOST_AUTO_TEST_CASE(test_nlopt_no_grad_simple)
{
    opt::NLOptGrad<Params, nlopt::LN_COBYLA> optimizer;
    Eigen::VectorXd g = optimizer(my_function, tools::random_vector(2), false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}
