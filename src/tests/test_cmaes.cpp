#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE nlopt_test

#include <boost/test/unit_test.hpp>

#include <limbo/limbo.hpp>
#include <limbo/opt/cmaes.hpp>

struct Params {
    struct opt_cmaes : public limbo::defaults::opt_cmaes {
    };
};

struct TestOpt {
public:
    double utility(const Eigen::VectorXd& params) const
    {
        return -params(0) * params(0) - params(1) * params(1);
    }

    Eigen::VectorXd init() const
    {
      return Eigen::VectorXd::Zero(param_size());
    }

    size_t param_size() const
    {
          return 2;
    }
};

BOOST_AUTO_TEST_CASE(test_cmaes_unbounded)
{
    TestOpt util;
    Eigen::VectorXd g = limbo::opt::Cmaes<Params>()(util, false);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}


BOOST_AUTO_TEST_CASE(test_cmaes_bounded)
{
    TestOpt util;
    Eigen::VectorXd g = limbo::opt::Cmaes<Params>()(util, true);

    BOOST_CHECK_SMALL(g(0), 0.00000001);
    BOOST_CHECK_SMALL(g(1), 0.00000001);
}
