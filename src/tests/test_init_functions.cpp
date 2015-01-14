#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE init_functions

#include <boost/test/unit_test.hpp>
#include "limbo/limbo.hpp"
#include "limbo/inner_cmaes.hpp"

using namespace limbo;


struct Params {
  struct boptimizer {
    BO_PARAM(double, noise, 0.01);
    BO_PARAM(int, dump_period, -1);
  };
  struct maxiterations {
    BO_PARAM(int, n_iterations, 0);
  };
  struct kf_maternfivehalfs {
    BO_PARAM(double, sigma, 1);
    BO_PARAM(double, l, 0.25);
  };
  struct ucb : public defaults::ucb {};
  struct gp_ucb : public defaults::gp_ucb {};
  struct gp_auto : public defaults::gp_auto {};
  struct meanconstant : public defaults::meanconstant {};
  struct cmaes : public defaults::cmaes {};
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


BOOST_AUTO_TEST_CASE(no_init) {
  typedef init_functions::NoInit<Params> Init_t;
  typedef BOptimizer<Params, init_fun<Init_t> > Opt_t;

  Opt_t opt;
  opt.optimize(fit_eval());
  BOOST_CHECK(opt.observations().size() == 1);
  BOOST_CHECK(opt.samples().size() == 1);
}


BOOST_AUTO_TEST_CASE(random_sampling) {

  struct MyParams : public Params {
    struct init {
      BO_PARAM(int, nb_samples, 10);
    };
  };

  typedef init_functions::RandomSampling<MyParams> Init_t;
  typedef BOptimizer<MyParams, init_fun<Init_t> > Opt_t;

  Opt_t opt;
  opt.optimize(fit_eval());
  BOOST_CHECK(opt.observations().size() == 11);
  BOOST_CHECK(opt.samples().size() == 11);
  for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
    const Eigen::VectorXd& x = opt.samples()[j];
    std::cout << x.transpose() << std::endl;
    for (int i = 0; i < x.size(); ++i) {
      BOOST_CHECK(x[i] >= 0);
      BOOST_CHECK(x[i] <= 1);
    }
  }
}

BOOST_AUTO_TEST_CASE(random_sampling_grid) {
  std::cout << "RandomSamplingGrid" << std::endl;
  struct MyParams : public Params {
    struct init {
      BO_PARAM(int, nb_samples, 10);
      BO_PARAM(int, nb_bins, 4);

    };
  };

  typedef init_functions::RandomSamplingGrid<MyParams> Init_t;
  typedef BOptimizer<MyParams, init_fun<Init_t> > Opt_t;

  Opt_t opt;
  opt.optimize(fit_eval());
  BOOST_CHECK(opt.observations().size() == 10 + 1);
  BOOST_CHECK(opt.samples().size() == 10 + 1);
  for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
    const Eigen::VectorXd& x = opt.samples()[j];
    std::cout << x.transpose() << std::endl;
    for (int i = 0; i < x.size(); ++i) {
      BOOST_CHECK(x[i] >= 0);
      BOOST_CHECK(x[i] <= 1);
      BOOST_CHECK(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
    }
  }
}



BOOST_AUTO_TEST_CASE(grid_sampling) {
  std::cout << "GirSampling" << std::endl;
  struct MyParams : public Params {
    struct init {
      BO_PARAM(int, nb_bins, 4);

    };
  };

  typedef init_functions::GridSampling<MyParams> Init_t;
  typedef BOptimizer<MyParams, init_fun<Init_t> > Opt_t;

  Opt_t opt;
  opt.optimize(fit_eval());
  std::cout << opt.observations().size() << std::endl;
  BOOST_CHECK(opt.observations().size() == 25 + 1);
  BOOST_CHECK(opt.samples().size() == 25 + 1);
  for (size_t j = 0; j < opt.samples().size() - 1; ++j) {
    const Eigen::VectorXd& x = opt.samples()[j];
    std::cout << x.transpose() << std::endl;
    for (int i = 0; i < x.size(); ++i) {
      BOOST_CHECK(x[i] >= 0);
      BOOST_CHECK(x[i] <= 1);
      BOOST_CHECK(x[i] == 0 || x[i] == 0.25 || x[i] == 0.5 || x[i] == 0.75 || x[i] == 1.0);
    }
  }
}
