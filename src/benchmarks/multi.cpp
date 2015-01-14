#include "limbo/limbo.hpp"
#include "limbo/parego.hpp"
#include "limbo/ehvi.hpp"
#include "limbo/nsbo.hpp"

using namespace limbo;

struct Params {
  struct boptimizer {
    BO_PARAM(double, noise, 0.01);
    BO_PARAM(int, dump_period, 1);
  };
  struct init {
    BO_PARAM(int, nb_samples, 10);
    // calandra: number of dimensions * 5
    // knowles : 11 * dim - 1
  };
  struct parego : public defaults::parego {};
  struct maxiterations {
    BO_PARAM(int, n_iterations, 30);
  };
  struct ucb : public defaults::ucb {};
  struct gp_ucb : public defaults::gp_ucb {};
  struct cmaes : public defaults::cmaes {};
  struct gp_auto : public defaults::gp_auto {};
  struct meanconstant : public defaults::meanconstant {};
  struct ehvi {
    BO_PARAM(double, x_ref, -11);
    BO_PARAM(double, y_ref, -11);
  };
};


#ifdef DIM6
#define ZDT_DIM 6
#elif defined (DIM2)
#define ZDT_DIM 2
#else
#define ZDT_DIM 30
#endif

struct zdt1 {
  static constexpr size_t dim = ZDT_DIM;
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    Eigen::VectorXd res(2);
    double f1 = x(0);
    double g = 1.0;
    for (int i = 1; i < x.size(); ++i)
      g += 9.0 / (x.size() - 1) * x(i);
    double h = 1.0f - sqrtf(f1 / g);
    double f2 = g * h;
    res(0) = 1.0 - f1;
    res(1) = 1.0 - f2;
    return res;
  }
};

struct zdt2 {
  static constexpr size_t dim = ZDT_DIM;
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    Eigen::VectorXd res(2);
    double f1 = x(0);
    double g = 1.0;
    for (int i = 1; i < x.size(); ++i)
      g += 9.0 / (x.size() - 1) * x(i);
    double h = 1.0f - pow((f1 / g), 2.0);
    double f2 = g * h;
    res(0) = 1.0 - f1;
    res(1) = 1.0 - f2;
    return res;
  }
};

struct zdt3 {
  static constexpr size_t dim = ZDT_DIM;
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    Eigen::VectorXd res(2);
    double f1 = x(0);
    double g = 1.0;
    for (int i = 1; i < x.size(); ++i)
      g += 9.0 / (x.size() - 1) * x(i);
    double h = 1.0f - sqrtf(f1 / g) - f1 / g * sin(10 * M_PI * f1);
    double f2 = g * h;
    res(0) = 1.0 - f1;
    res(1) = 1.0 - f2;
    return res;
  }
};


struct mop2 {
  static constexpr size_t dim = 2;
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
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


namespace limbo {
  namespace stat {
    template<typename F>
    struct ParetoBenchmark {
      template<typename BO>
      void operator()(BO& opt) {
        opt.update_pareto_data();
#ifndef NSBO // this is already done is NSBO
        opt.template update_pareto_model<F::dim>();
#endif
        auto dir = opt.res_dir() + "/";
        auto p_model = opt.pareto_model();
        auto p_data = opt.pareto_data();
        std::string it = std::to_string(opt.iteration());
        std::string model = dir + "pareto_model_" + it + ".dat";
        std::string model_real = dir + "pareto_model_real_" + it + ".dat";
        std::string data = dir + "pareto_data_" + it + ".dat";
        std::string obs_f = dir + "obs_" + it + ".dat";
        std::ofstream pareto_model(model.c_str()),
            pareto_data(data.c_str()),
            pareto_model_real(model_real.c_str()),
            obs(obs_f.c_str());
        F f;
        for (auto x : p_model)
          pareto_model << std::get<1>(x).transpose() << " "
                       << std::get<2>(x).transpose()
                       << std::endl;
        for (auto x : p_model)
          pareto_model_real << f(std::get<0>(x)).transpose() << " "
                            << std::endl;
        for (auto x : p_data)
          pareto_data << std::get<1>(x).transpose() << std::endl;
        for (size_t i = 0; i < opt.observations().size(); ++i)
          obs << opt.observations()[i].transpose() << " "
              << opt.samples()[i].transpose()
              << std::endl;
        /*
                std::string m1 = "model_" + it + ".dat";
                std::ofstream m1f(m1.c_str());
                for (float x = 0; x < 1; x += 0.01)
                  for (float y = 0; y < 1; y += 0.01) {
                    Eigen::VectorXd v(2);
                    v << x, y;
                    m1f << x << " " << y << " "
                        << opt.models()[0].mu(v) << " "
                        << opt.models()[0].sigma(v) << " "
                        << opt.models()[1].mu(v) << " "
                        << opt.models()[1].sigma(v) << std::endl;

                  }
        */
        std::cout << "stats done" << std::endl;
      }
    };
  }


}

int main() {
  par::init();

#ifdef ZDT1
  typedef zdt1 func_t;
#elif defined ZDT2
  typedef zdt2 func_t;
#elif defined ZDT3
  typedef zdt3 func_t;
#elif defined MOP2
  typedef mop2 func_t;
#else
  typedef mop2 func_t;
#endif

  typedef stat::ParetoBenchmark<func_t> stat_t;
#ifdef PAREGO
  Parego<Params, stat_fun<stat_t> > opt;
#elif defined(NSBO)
  Nsbo<Params, stat_fun<stat_t> > opt;
#else
  Ehvi<Params, stat_fun<stat_t> > opt;
#endif

  opt.optimize(func_t());
  return 0;
}
