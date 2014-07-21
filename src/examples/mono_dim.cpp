//#define SHOW_TIMER
#include <boost/fusion/container/vector.hpp>
#include <boost/fusion/include/vector.hpp>

#include "limbo/limbo.hpp"
#include "limbo/inner_cmaes.hpp"

using namespace limbo;

struct Params {
  struct gp_ucb : public defaults::gp_ucb {};
  struct cmaes : public defaults::cmaes {};
  struct gp_auto : public defaults::gp_auto {};
  struct meanconstant : public defaults::meanconstant {};

  struct boptimizer {
    BO_PARAM(double, noise, 0.001);
    BO_PARAM(int, dump_period, 1);
  };
  struct init {
    BO_PARAM(int, nb_samples, 5);
  };
  struct maxiterations {
    BO_PARAM(int, n_iterations, 20);
  };

};

struct fit_eval {
  static constexpr size_t dim = 2;
  double operator()(const Eigen::VectorXd& x) const {
    double res = 0;
    for (int i = 0; i < x.size(); i++)
      res += 1 - (x[i] - 0.3) * (x[i] - 0.3) + sin(10 * x[i]) * 0.2;
    return res;
  }

};


int main() {
  BOptimizer<Params> opt;
  opt.optimize(fit_eval());
  std::cout << opt.best_observation()
            << " res  " << opt.best_sample().transpose()
            << std::endl;
  return 0;
}
