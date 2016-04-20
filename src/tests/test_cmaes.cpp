#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_cmaes

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
    size_t N = 10;
    size_t errors = 0;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = limbo::opt::Cmaes<Params>()(fsphere, Eigen::VectorXd::Zero(2), false);

        if (std::abs(g(0)) > 0.00000001 || std::abs(g(1)) > 0.00000001)
            errors++;
    }

    BOOST_CHECK((double(errors) / double(N)) < 0.4);
}

BOOST_AUTO_TEST_CASE(test_cmaes_bounded)
{
    size_t N = 10;
    size_t errors = 0;
    for (size_t i = 0; i < N; i++) {
        Eigen::VectorXd g = limbo::opt::Cmaes<Params>()(fsphere, Eigen::VectorXd::Zero(2), true);

        if (std::abs(g(0)) > 0.00000001 || std::abs(g(1)) > 0.00000001)
            errors++;
    }

    BOOST_CHECK((double(errors) / double(N)) < 0.3);
}
