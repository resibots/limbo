#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE optimizers

#include <boost/test/unit_test.hpp>

#include <limbo/tools/macros.hpp>
#include <limbo/opt/cmaes.hpp>
#include <limbo/opt/grid_search.hpp>
#include <limbo/opt/random_point.hpp>

using namespace limbo;

Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
}

struct Params {
    struct grid_search {
        BO_PARAM(int, nb_pts, 20);
    };

    struct cmaes : public defaults::cmaes {
    };
};

int monodim_calls = 0;

struct FakeAcquiMono {
    size_t dim_in() const { return 1; }

    double operator()(const Eigen::VectorXd& v) const
    {
        monodim_calls++;
        return 3 * v(0) + 5;
    }
};

int bidim_calls = 0;

struct FakeAcquiBi {
    size_t dim_in() const { return 2; }

    double operator()(const Eigen::VectorXd& v) const
    {
        bidim_calls++;
        return 3 * v(0) + 5 - 2 * v(1) - 5 * v(1) + 2;
    }
};

template <typename Functor>
struct FunctorOptimization {
public:
    FunctorOptimization(const Functor& f, const Eigen::VectorXd& init) : _f(f), _init(init) {}

    double utility(const Eigen::VectorXd& params) const
    {
        return _f(params);
    }

    size_t param_size() const
    {
        return _f.dim_in();
    }

    const Eigen::VectorXd& init() const
    {
        return _init;
    }

protected:
    const Functor& _f;
    const Eigen::VectorXd& _init;
};

template <typename Functor>
FunctorOptimization<Functor> make_functor_optimization(const Functor& f)
{
    return FunctorOptimization<Functor>(f, Eigen::VectorXd::Constant(f.dim_in(), 0.5));
}

BOOST_AUTO_TEST_CASE(test_random_mono_dim)
{
    using namespace limbo;

    opt::RandomPoint<Params> optimizer;

    FakeAcquiMono f;
    monodim_calls = 0;
    for (int i = 0; i < 1000; i++) {
        Eigen::VectorXd best_point = optimizer(make_functor_optimization(f));
        BOOST_CHECK_EQUAL(best_point.size(), 1);
        BOOST_CHECK(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
        BOOST_CHECK(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(test_random_bi_dim)
{
    using namespace limbo;

    opt::RandomPoint<Params> optimizer;

    FakeAcquiBi f;
    auto f_optimization = make_functor_optimization(f);
    bidim_calls = 0;
    for (int i = 0; i < 1000; i++) {
        Eigen::VectorXd best_point = optimizer(f_optimization);
        BOOST_CHECK_EQUAL(best_point.size(), 2);
        BOOST_CHECK(best_point(0) > 0 || std::abs(best_point(0)) < 1e-7);
        BOOST_CHECK(best_point(0) < 1 || std::abs(best_point(0) - 1) < 1e-7);
        BOOST_CHECK(best_point(1) > 0 || std::abs(best_point(1)) < 1e-7);
        BOOST_CHECK(best_point(1) < 1 || std::abs(best_point(1) - 1) < 1e-7);
    }
}

BOOST_AUTO_TEST_CASE(test_grid_search_mono_dim)
{
    using namespace limbo;

    opt::GridSearch<Params> optimizer;

    FakeAcquiMono f;
    monodim_calls = 0;
    Eigen::VectorXd best_point = optimizer(make_functor_optimization(f));

    BOOST_CHECK_EQUAL(best_point.size(), 1);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_EQUAL(monodim_calls, Params::grid_search::nb_pts() + 1);
}

BOOST_AUTO_TEST_CASE(test_grid_search_bi_dim)
{
    using namespace limbo;

    opt::GridSearch<Params> optimizer;

    FakeAcquiBi f;
    auto f_optimization = make_functor_optimization(f);
    bidim_calls = 0;
    Eigen::VectorXd best_point = optimizer(f_optimization);

    BOOST_CHECK_EQUAL(best_point.size(), 2);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_SMALL(best_point(1), 0.000001);
    // TO-DO: Maybe alter a little grid search so not to call more times the utility function
    BOOST_CHECK_EQUAL(bidim_calls, (Params::grid_search::nb_pts() + 1) * (Params::grid_search::nb_pts() + 1) + 21);
}

BOOST_AUTO_TEST_CASE(test_cmaes_mono_dim)
{
    using namespace limbo;

    opt::Cmaes<Params> optimizer;

    FakeAcquiMono f;
    Eigen::VectorXd best_point = optimizer(make_functor_optimization(f));

    BOOST_CHECK_EQUAL(best_point.size(), 1);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
}

BOOST_AUTO_TEST_CASE(test_cmaes_bi_dim)
{
    using namespace limbo;

    opt::Cmaes<Params> optimizer;

    FakeAcquiBi f;
    auto f_optimization = make_functor_optimization(f);
    Eigen::VectorXd best_point = optimizer(f_optimization);

    BOOST_CHECK_EQUAL(best_point.size(), 2);
    BOOST_CHECK_CLOSE(best_point(0), 1, 0.0001);
    BOOST_CHECK_SMALL(best_point(1), 0.000001);
}