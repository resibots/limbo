#include <limbo/tools/macros.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/function_ard.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/kernel_mean_lf_opt.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>

using namespace limbo;

struct Params {
#ifdef USE_LIBCMAES
    struct opt_cmaes : public defaults::opt_cmaes {
    };
#elif defined(USE_NLOPT)
    struct opt_nloptnograd : public defaults::opt_nloptnograd {
    };
#else
    struct opt_gridsearch : public defaults::opt_gridsearch {
    };
#endif
    struct opt_rprop : public defaults::opt_rprop {
    };

    struct kernel_maternfivehalfs {
        BO_PARAM(double, sigma, 1);
        BO_PARAM(double, l, 0.2);
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.001);
        BO_PARAM(bool, stats_enabled, true);
    };

    struct init_randomsampling {
        BO_PARAM(int, samples, 5);
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 100);
    };

    struct opt_parallelrepeater : defaults::opt_parallelrepeater {
    };
};

template <typename Params, typename Model>
class UCB_multi {
public:
    UCB_multi(const Model& model, int iteration = 0) : _model(model) {}

    size_t dim_in() const { return _model.dim_in(); }

    size_t dim_out() const { return _model.dim_out(); }

    template <typename AggregatorFunction>
    double operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun) const
    {
        // double mu, sigma;
        // std::tie(mu, sigma) = _model.query(v);
        // return (mu + Params::ucb::alpha() * sqrt(sigma));

        return (sqrt(_model.sigma(v)));
    }

protected:
    const Model& _model;
};

template <typename Params>
struct MeanOffset {
    MeanOffset(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2; // constant overestimation
        res(1) = 2; // constant overestimation

        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        return res;
    }
};

template <typename Params>
struct MeanRotation {
    MeanRotation(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
    {
        Eigen::VectorXd res(2);
        res(0) = 0; // constant overestimation
        res(1) = 0; // constant overestimation
        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        double theta = M_PI / 2;
        Eigen::Matrix2d rot;
        rot(0, 0) = cos(theta);
        rot(0, 1) = -sin(theta);
        rot(1, 0) = sin(theta);
        rot(1, 1) = cos(theta);
        return rot * res;
    }
};

template <typename Params>
struct MeanComplet {
    MeanComplet(size_t dim_out = 1) {}

    template <typename GP>
    Eigen::VectorXd operator()(const Eigen::VectorXd& x, const GP& gp) const
    {
        Eigen::VectorXd res(2);
        res(0) = 2; // constant overestimation
        res(1) = 2; // constant overestimation
        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        double theta = M_PI / 2;
        Eigen::Matrix2d rot;
        rot(0, 0) = cos(theta);
        rot(0, 1) = -sin(theta);
        rot(1, 0) = sin(theta);
        rot(1, 1) = cos(theta);
        return rot * res;
    }
};

struct fit_eval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(dim_out);
        res(0) = 0;
        res(1) = 0;
        for (int i = 0; i < x.size(); i++) {
            res(0) += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
            res(1) += 1 - (x[i] - 0.3) * (x[i] - 0.3) * 0.4;
        }
        return res;
    }
};

int main()
{

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef mean::FunctionARD<Params, MeanComplet<Params>> Mean_t;
    typedef model::GP<Params, Kernel_t, Mean_t, model::gp::KernelMeanLFOpt<Params>> GP_t;
    typedef UCB_multi<Params, GP_t> Acqui_t;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, acquifun<Acqui_t>> opt;
    opt.optimize(fit_eval());

    std::cout << opt.best_observation() << " res  "
              << opt.best_sample().transpose() << std::endl;
    return 0;
}
