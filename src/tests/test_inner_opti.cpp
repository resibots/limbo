#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE inner_optimization

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/inner_opt.hpp>
#include <limbo/bayes_opt/bo_base.hpp>

using namespace limbo;

Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
}

struct Params {
    struct exhaustive_search {
        BO_PARAM(int, nb_pts, 20);
    };

    struct cmaes : public defaults::cmaes {
    };
};

int monodim_calls = 0;

struct FakeAcquiMono {
    size_t dim_in() const { return 1; }

    template <typename AggregatorFunction>
    double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
    {
        monodim_calls++;
        return 3 * afun(v) + 5;
    }
};

int bidim_calls = 0;

struct FakeAcquiBi {
    size_t dim_in() const { return 2; }

    template <typename AggregatorFunction>
    double operator()(const Eigen::VectorXd& v, const AggregatorFunction&) const
    {
        bidim_calls++;
        return 3 * v(0) + 5 - 2 * v(1) - 5 * v(1) + 2;
    }
};

BOOST_AUTO_TEST_CASE(test_random_mono_dim)
{
    using namespace limbo;

    inner_opt::Random<Params> inner_optimization;

    FakeAcquiMono f;
    monodim_calls = 0;
    for (int i = 0; i < 1000; i++) {
        Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());
        BOOST_CHECK_EQUAL(best_point.size(), 1);
        BOOST_CHECK(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
        BOOST_CHECK(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(test_random_bi_dim)
{
    using namespace limbo;

    inner_opt::ExhaustiveSearch<Params> inner_optimization;

    FakeAcquiBi f;
    bidim_calls = 0;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    for (int i = 0; i < 1000; i++) {
        Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());
        BOOST_CHECK_EQUAL(best_point.size(), 2);
        BOOST_CHECK(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
        BOOST_CHECK(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
        BOOST_CHECK(best_point(1) > 0 || std::abs(best_point(1)) < 1e-7);
        BOOST_CHECK(best_point(1) < 1 || std::abs(best_point(1) - 1) < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(test_exhaustive_search_mono_dim)
{
    using namespace limbo;

    inner_opt::ExhaustiveSearch<Params> inner_optimization;

    FakeAcquiMono f;
    monodim_calls = 0;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    BOOST_CHECK_EQUAL(best_point.size(), 1);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_EQUAL(monodim_calls, Params::exhaustive_search::nb_pts() + 1);
}

BOOST_AUTO_TEST_CASE(test_exhaustive_search_bi_dim)
{
    using namespace limbo;

    inner_opt::ExhaustiveSearch<Params> inner_optimization;

    FakeAcquiBi f;
    bidim_calls = 0;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    BOOST_CHECK_EQUAL(best_point.size(), 2);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_SMALL(best_point(1), 0.000001);
    BOOST_CHECK_EQUAL(bidim_calls, (Params::exhaustive_search::nb_pts() + 1) * (Params::exhaustive_search::nb_pts() + 1));
}

BOOST_AUTO_TEST_CASE(test_cmaes_mono_dim)
{
    using namespace limbo;

    Cmaes<Params> inner_optimization;

    FakeAcquiMono f;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    BOOST_CHECK_EQUAL(best_point.size(), 1);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
}

BOOST_AUTO_TEST_CASE(test_cmaes_bi_dim)
{
    using namespace limbo;

    Cmaes<Params> inner_optimization;

    FakeAcquiBi f;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    BOOST_CHECK_EQUAL(best_point.size(), 2);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_SMALL(best_point(1), 0.000001);
}