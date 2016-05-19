#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE test_kernel

#include <boost/test/unit_test.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <iostream>

using namespace limbo;
struct Params {
    struct kernel_squared_exp_ard {
        BO_DYN_PARAM(int, k); //equivalent to the standard exp ARD
    };
};

BO_DECLARE_DYN_PARAM(int, Params::kernel_squared_exp_ard, k);

Eigen::VectorXd make_v2(double x1, double x2)
{
    Eigen::VectorXd v2(2);
    v2 << x1, x2;
    return v2;
}

BOOST_AUTO_TEST_CASE(test_kernel_SE_ARD)
{
    Params::kernel_squared_exp_ard::set_k(0);

    kernel::SquaredExpARD<Params> se(2);
    Eigen::VectorXd hp(se.h_params_size());
    hp(0) = 0; //exp(0)=1
    hp(1) = 0;
    hp(2) = 1;
    se.set_h_params(hp);

    Eigen::VectorXd v1 = make_v2(1, 1);
    BOOST_CHECK(se(v1, v1) == 1);

    Eigen::VectorXd v2 = make_v2(0, 1);
    double s1 = se(v1, v2);

    BOOST_CHECK(std::abs(s1 - exp(-0.5 * (v1.transpose() * v2)[0])) < 1e-5);

    hp(0) = 1;
    se.set_h_params(hp);
    double s2 = se(v1, v2);
    BOOST_CHECK(s1 < s2);

    Params::kernel_squared_exp_ard::set_k(1);
    se = kernel::SquaredExpARD<Params>(2);
    hp = Eigen::VectorXd(se.h_params_size());
    hp(0) = 0;
    hp(1) = 0;
    hp(2) = 0;
    hp(3) = 0;
    hp(4) = 1;

    se.set_h_params(hp);
    BOOST_CHECK(s1 == se(v1, v2));
}
