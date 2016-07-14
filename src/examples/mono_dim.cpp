#include <limbo/tools/macros.hpp>
#include <limbo/kernel/matern_five_halfs.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/acqui/gp_ucb.hpp>
#include <limbo/bayes_opt/boptimizer.hpp>
#include <limbo/stat.hpp>

using namespace limbo;

BO_PARAMS(std::cout,
          struct Params {
              struct acqui_gpucb : public defaults::acqui_gpucb {
              };

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
              struct acqui_ucb {
                  BO_PARAM(double, alpha, 0.1);
              };

              struct kernel_maternfivehalfs {
                  BO_PARAM(double, sigma_sq, 1);
                  BO_PARAM(double, l, 0.2);
              };

              struct bayes_opt_bobase {
                BO_PARAM(bool, stats_enabled, true);
              };

              struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
                  BO_PARAM(double, noise, 0.001);
              };

              struct init_randomsampling {
                  BO_PARAM(int, samples, 5);
              };

              struct stop_maxiterations {
                  BO_PARAM(int, iterations, 20);
              };
              struct stat_gp {
                BO_PARAM(int, bins, 20);
              };
          };)

struct fit_eval {
    static constexpr size_t dim_in = 2;
    static constexpr size_t dim_out = 1;

    Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
    {
        double res = 0;
        for (int i = 0; i < x.size(); i++)
            res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
        return tools::make_vector(res);
    }
};

int main()
{
    typedef kernel::MaternFiveHalves<Params> Kernel_t;
    typedef mean::Data<Params> Mean_t;
    typedef model::GP<Params, Kernel_t, Mean_t> GP_t;
    typedef acqui::UCB<Params, GP_t> Acqui_t;
    using stat_t = boost::fusion::vector<stat::ConsoleSummary<Params>,
        stat::Samples<Params>,
        stat::Observations<Params>,
        stat::GP<Params>>;

    bayes_opt::BOptimizer<Params, modelfun<GP_t>, statsfun<stat_t>, acquifun<Acqui_t>> opt;
    opt.optimize(fit_eval());
    std::cout << opt.best_observation() << " res  "
              << opt.best_sample().transpose() << std::endl;
    return 0;
}
