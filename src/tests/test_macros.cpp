#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE macros

#include <cstring>

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>

struct Params {
    struct test {
        BO_PARAM(double, a, 1);
        BO_DYN_PARAM(int, b);
        BO_PARAM_ARRAY(double, c, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        BO_PARAM_VECTOR(double, d, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        BO_PARAM_STRING(e, "e");
    };
};

BO_DECLARE_DYN_PARAM(int, Params::test, b);

BOOST_AUTO_TEST_CASE(test_macros)
{
    BOOST_CHECK(Params::test::a() == 1.0);

    Params::test::set_b(2);
    BOOST_CHECK(Params::test::b() == 2);
    Params::test::set_b(3);
    BOOST_CHECK(Params::test::b() == 3);

    BOOST_CHECK(__VA_NARG__(1) == 1);
    BOOST_CHECK(__VA_NARG__(10, 11, 12, 13) == 4);

    BOOST_CHECK(Params::test::c_size() == 6);
    BOOST_CHECK(Params::test::c(0) == 1.0);
    BOOST_CHECK(Params::test::c(1) == 2.0);
    BOOST_CHECK(Params::test::c(2) == 3.0);
    BOOST_CHECK(Params::test::c(3) == 4.0);
    BOOST_CHECK(Params::test::c(4) == 5.0);
    BOOST_CHECK(Params::test::c(5) == 6.0);

    BOOST_CHECK(Params::test::d().size() == 6);
    BOOST_CHECK(Params::test::d()(0) == 1.0);
    BOOST_CHECK(Params::test::d()(1) == 2.0);
    BOOST_CHECK(Params::test::d()(2) == 3.0);
    BOOST_CHECK(Params::test::d()(3) == 4.0);
    BOOST_CHECK(Params::test::d()(4) == 5.0);
    BOOST_CHECK(Params::test::d()(5) == 6.0);

    BOOST_CHECK(strcmp(Params::test::e(), "e") == 0);
}
