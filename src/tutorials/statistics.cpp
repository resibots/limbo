// please see the explanation in the documentation

#include <iostream>
#include <limbo/limbo.hpp>

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

template <typename Params>
struct WorstObservation : public stat::StatBase<Params> {
    template <typename BO, typename AggregatorFunction>
    void operator()(const BO& bo, const AggregatorFunction& afun, bool blacklisted)
    {
      // [optional] if statistics have been disabled or if there are no observations, we do not do anything
      if (!bo.stats_enabled() || bo.observations().empty())
          return;

      // [optional] we create a file to write / you can use your own file but remember that this method is called at each iteration (you need to create it in the constructor)
      this->_create_log_file(bo, "worst_observations.dat");

      // [optional] we add a header to the file to make it easier to read later
      if (bo.total_iterations() == 0)
          (*this->_log_file) << "#iteration worst_observation sample" << std::endl;

      // ----- search for the worst observation ----
      // 1. get the aggregated observations
      auto rewards = std::vector<double>(bo.observations().size());
      std::transform(bo.observations().begin(), bo.observations().end(), rewards.begin(), afun);
      // 2. search for the worst element
      auto min_e = std::min_element(rewards.begin(), rewards.end());
      auto min_obs = bo.observations()[std::distance(rewards.begin(), min_e)];
      auto min_sample = bo.samples()[std::distance(rewards.begin(), min_e)];

      // ----- write what we have found ------
      // the file is (*this->_log_file)
      (*this->_log_file) << bo.total_iterations() << " " << min_obs.transpose() << " " << min_sample.transpose() << std::endl;
    }
};

int main() {
  // we use the default acquisition function / model / stat / etc.

  // define a special list of statistics which include our new statistics class
  using stat_t =
    boost::fusion::vector<stat::ConsoleSummary<Params>,
                          stat::Samples<Params>,
                          stat::Observations<Params>,
                          WorstObservation<Params> >;

  /// remmeber to use the new statistics vector via statsfun<>!
  bayes_opt::BOptimizer<Params, statsfun<stat_t>> boptimizer;

  // run the evaluation
  boptimizer.optimize(Eval());

  // the best sample found
  std::cout << "Best sample: " << boptimizer.best_sample()(0) << " - Best observation: " << boptimizer.best_observation()(0) << std::endl;
  return 0;
}
