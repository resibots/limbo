// please see the explanation in the documentation

#include <iostream>

// you can also include <limbo/limbo.hpp> but it will slow down the compilation
#include <limbo/bayes_opt/boptimizer.hpp>

using namespace limbo;

struct Params {
  // no noise
  struct bayes_opt_boptimizer {
    BO_PARAM(double, noise, 0.0);
  };

  // depending on which internal optimizer we use, we need to import different parameters
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

  // enable / disable the writing of the result files
  struct bayes_opt_bobase {
    BO_PARAM(int, stats_enabled, true);
  };

  // we use 10 random samples to initialize the algorithm
  struct init_randomsampling {
    BO_PARAM(int, samples, 10);
  };

  // we stop after 40 iterations
  struct stop_maxiterations {
    BO_PARAM(int, iterations, 40);
  };

  // we use the default parameters for acqui_ucb
  struct acqui_gpucb : public defaults::acqui_gpucb {
  };

  /// we use the default parameters for rprop (hyper-parameter optimization)
  struct opt_rprop : public defaults::opt_rprop {
  };

  /// default parameters for the parallel_repeater ((hyper-parameter optimization))
  struct opt_parallelrepeater : public defaults::opt_parallelrepeater {
  };
};

struct Eval {
  // number of input dimension (x.size())
  static constexpr size_t dim_in = 1;
  // number of dimenions of the result (res.size())
  static constexpr size_t dim_out = 1;

  // the function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const
  {
    double y = -((5 * x(0) - 2.5) * (5 * x(0) - 2.5)) + 5;
    // we return a 1-dimensional vector
    return tools::make_vector(y);
  }
};

int main() {
  // we use the default acquisition function / model / stat / etc.
  bayes_opt::BOptimizer<Params> boptimizer;
  // run the evaluation
  boptimizer.optimize(Eval());
  // the best sample found
  std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
  return 0;
}
