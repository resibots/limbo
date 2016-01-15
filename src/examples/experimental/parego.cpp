#include <limbo/limbo.hpp>
#include <limbo/experimental/bayes_opt/parego.hpp>

using namespace limbo;

struct Params {
    struct init_randomsampling {
        BO_PARAM(int, samples, 21);
        // calandra: number of dimensions * 5
        // knowles : 11 * dim - 1
    };

    struct stop_maxiterations {
        BO_PARAM(int, iterations, 30);
    };

    struct acqui_ucb : public defaults::acqui_ucb {
    };

    struct acqui_gpucb : public defaults::acqui_gpucb {
    };

    struct opt_cmaes : public defaults::opt_cmaes {
    };

    struct mean_constant : public defaults::mean_constant {
    };

    struct bayes_opt_bobase {
        BO_PARAM(bool, stats_enabled, true);
    };

    struct bayes_opt_parego : public defaults::bayes_opt_parego {
        BO_PARAM(double, noise, 0.005);
    };
};

struct zdt2 {
    static constexpr size_t dim_in = 30;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        double f1 = x(0);
        double g = 1.0;
        for (int i = 1; i < x.size(); ++i)
            g += 9.0 / (x.size() - 1) * x(i) * x(i);
        double h = 1.0f - pow((f1 / g), 2.0);
        double f2 = g * h;
        res(0) = -f1;
        res(1) = -f2;
        return res;
    }
};

struct mop2 {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 2;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd res(2);
        // scale to [-2, 2]
        Eigen::VectorXd xx = (x * 4.0).array() - 2.0;
        // f1, f2
        Eigen::VectorXd v1 = (xx.array() - 1.0 / sqrt(xx.size())).array().square();
        Eigen::VectorXd v2 = (xx.array() + 1.0 / sqrt(xx.size())).array().square();
        double f1 = 1.0 - exp(-v1.sum());
        double f2 = 1.0 - exp(-v2.sum());
        // we _maximize in [0:1]
        res(0) = -f1 + 1;
        res(1) = -f2 + 1;
        return res;
    }
};

int main()
{
    tools::par::init();
    // if you want to use a standard GP & basic UCB:
    // typedef kernel_functions::MaternFiveHalfs<Params> kernel_t;
    // typedef model::GP<Params, kernel_t, mean_t> gp_t;
    // typedef acquisition_functions::UCB<Params, gp_t> ucb_t;
    // Parego<Params, model_fun<gp_t>, acq_fun<ucb_t> > opt;
    experimental::bayes_opt::Parego<Params> opt;
    opt.optimize(mop2());

    std::cout << "optimization done" << std::endl;
    auto p_model = opt.pareto_model();
    auto p_data = opt.pareto_data();

    std::ofstream pareto_model("mop2_pareto_model.dat"),
        pareto_data("mop2_pareto_data.dat");
    std::cout << "writing..." << std::endl;
    for (auto x : p_model)
        pareto_model << std::get<1>(x).transpose() << " " << std::endl;
    for (auto x : p_data)
        pareto_data << std::get<1>(x).transpose() << std::endl;

    return 0;
}
