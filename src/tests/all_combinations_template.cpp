#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#include <limbo/limbo.hpp>

using namespace limbo;

struct Params {

    struct kf_exp {
        BO_PARAM(double, sigma, 1);
    };

    struct kf_maternthreehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct kf_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct meanconstant {
        BO_PARAM_VECTOR(double, constant, 0, 0);
    };

    struct gp_auto : defaults::gp_auto {
    };

    struct gp_auto_mean : defaults::gp_auto_mean {
    };

    struct ucb : public defaults::ucb {
    };

    struct gp_ucb : public defaults::gp_ucb {
    };

    struct exhaustive_search {
        BO_PARAM(int, nb_pts, 20);
    };

    struct cmaes : public defaults::cmaes {
    };

    struct init {
        BO_PARAM(int, nb_samples, 5);
        BO_PARAM(int, nb_bins, 5);
    };

    struct maxiterations {
        BO_PARAM(int, n_iterations, 20);
    };

    struct maxpredictedvalue {
        BO_PARAM(double, ratio, 2);
    };

    struct boptimizer {
        BO_PARAM(double, noise, 0.001);
        BO_PARAM(int, dump_period, 1);
    };
};

struct MeanEval {
    MeanEval(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP&) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2.5 * x(0);
        res(1) = -4.5 * x(1);
        return res;
    }
};

struct Average {
    typedef double result_type;
    double operator()(const Eigen::VectorXd& x) const
    {
        return (x(0) + x(1)) / 2;
    }
};

struct StateEval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        res(0) = 3 * x(0) + 5;
        res(1) = -5 * x(1) + 2;
        return res;
    }
};

int main()
{
    @declarations
    @optimizer.optimize(StateEval());
    @optimizer.best_observation();
    @optimizer.best_sample();
    @optimizer.optimize(StateEval(), Average(), true);
    @optimizer.best_observation(Average());
    @optimizer.best_sample(Average());
}