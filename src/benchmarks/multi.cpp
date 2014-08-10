#include "limbo/limbo.hpp"
#include "limbo/ns_ego.hpp"
#include "limbo/parego.hpp"

using namespace limbo;

struct Params {
  struct boptimizer {
    BO_PARAM(double, noise, 0.0);
    BO_PARAM(int, dump_period, 1);
  };
  struct init {
    BO_PARAM(int, nb_samples, 50);
    // calandra: number of dimensions * 5
    // knowles : 11 * dim - 1
  };
  struct parego : public defaults::parego {};
  struct maxiterations {
    BO_PARAM(int, n_iterations, 100);
  };
  struct ucb : public defaults::ucb {};
  struct gp_ucb : public defaults::gp_ucb {};
  struct cmaes : public defaults::cmaes {};
  struct gp_auto : public defaults::gp_auto {};
  struct meanconstant : public defaults::meanconstant {};
};


struct zdt2 {
  static constexpr size_t dim = 30;
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    Eigen::VectorXd res(2);
    double f1 = x(0);
    double g = 1.0;
    for (size_t i = 1; i < x.size(); ++i)
      g += 9.0 / (x.size() - 1) * x(i);
    double h = 1.0f - pow((f1 / g), 2.0);
    double f2 = g * h;
    res(0) = -f1 + 1;
    res(1) = -f2 + 1;
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
      void operator()(const BO& opt) {
        auto dir = opt.res_dir() + "/";
        auto p_model = opt.model_pareto_front();
        auto p_data = opt.data_pareto_front();
        std::string it = std::to_string(opt.iteration());
        std::string model = dir + "pareto_model_" + it + ".dat";
        std::string model_real = dir + "pareto_model_real_" + it + ".dat";
        std::string data = dir + "pareto_data_" + it + ".dat";
        std::ofstream pareto_model(model.c_str()),
            pareto_data(data.c_str()),
            pareto_model_real(model_real.c_str());
        F f;
        for (auto x : p_model)
          pareto_model << std::get<1>(x).transpose() << " "
                       << std::endl;
        for (auto x : p_model)
          pareto_model_real << f(std::get<0>(x)).transpose() << " "
                            << std::endl;
        for (auto x : p_data)
          pareto_data << std::get<1>(x).transpose() << std::endl;

      }
    };
  }


}

int main() {
  par::init();

#ifdef ZDT2
  typedef zdt2 func_t;
#elif defined MOP2
  typedef mop2 func_t;
#else
# error "unknown function to optimize"
#endif


#ifdef NS_EGO
  NsEgo<Params, stat_fun<stat::ParetoBenchmark<func_t> > > opt;
#else
  Parego<Params, stat_fun<stat::ParetoBenchmark<func_t> > > opt;
#endif

  opt.optimize(func_t());
  return 0;
}
