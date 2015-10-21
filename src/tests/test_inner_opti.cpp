#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE parallel

#include <boost/test/unit_test.hpp>

#include "limbo/limbo.hpp"

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

    struct meanconstant {
        static Eigen::VectorXd constant() { return make_v1(0.0); };
    };
};

int monodim_calls = 0;

struct FakeAcquiMono {
    size_t dim_in() const { return 1; }

    template <typename AggregatorFunction>    
    double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun)  const
    {
        monodim_calls++;
        return 3 * afun(v) + 5;
    }
};

BOOST_AUTO_TEST_CASE(test_exhaustive_search_mono_dim)
{
    using namespace limbo;

    typedef inner_optimization::ExhaustiveSearch<Params> ex_search_t;
    ex_search_t inner_optimization;

    FakeAcquiMono f;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_EQUAL(monodim_calls, Params::exhaustive_search::nb_pts() + 1);
}

int bidim_calls = 0;

struct FakeAcquiBi {
    size_t dim_in() const { return 2; }

    template <typename AggregatorFunction>    
    double operator()(const Eigen::VectorXd& v, const AggregatorFunction&)  const
    {
        bidim_calls++;
        return 3 * v(0) + 5 - 2 * v(1) -5 * v(1) + 2;
    }
};

BOOST_AUTO_TEST_CASE(test_exhaustive_search_bi_dim)
{
    using namespace limbo;

    typedef inner_optimization::ExhaustiveSearch<Params> ex_search_t;
    ex_search_t inner_optimization;

    FakeAcquiBi f;
    Eigen::VectorXd best_point = inner_optimization(f, f.dim_in(), FirstElem());

    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_CLOSE(best_point(1), 0, 0.0001);
    BOOST_CHECK_EQUAL(bidim_calls, (Params::exhaustive_search::nb_pts() + 1) * (Params::exhaustive_search::nb_pts() + 1));
}