#include "limbo/parego.hpp"

using namespace limbo;

struct Params {
  struct boptimizer {
    BO_PARAM(double, noise, 0.005);
    BO_PARAM(int, dump_period, -1);
  };
  struct init {
    BO_PARAM(int, nb_samples, 21);
    // calandra: number of dimensions * 5
    // knowles : 11 * dim - 1
  };
  struct maxiterations {
    BO_PARAM(int, n_iterations, 30);
  };
  struct ucb : public defaults::ucb {};
  struct gp_ucb : public defaults::gp_ucb {};
  struct cmaes : public defaults::cmaes {};
  struct gp_auto : public defaults::gp_auto {};
  struct meanconstant : public defaults::meanconstant {};
  struct parego : public defaults::parego {};
};


struct zdt2 {
  static constexpr size_t dim = 30;
  Eigen::VectorXd operator()(const Eigen::VectorXd& x) const {
    Eigen::VectorXd res(2);
    double f1 = x(0);
    double g = 1.0;
    for (size_t i = 1; i < x.size(); ++i)
      g += 9.0 / (x.size() - 1) * x(i) * x(i);
    double h = 1.0f - pow((f1 / g), 2.0);
    double f2 = g * h;
    res(0) = -f1;
    res(1) = -f2;
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

struct Pareto {
  template<typename BO>
  void operator()(const BO& opt) {
    if (opt.iteration() % 10 != 0)
      return;
    auto p_model = opt.model_pareto_front(0, 1.0, 0.005);
    auto p_data = opt.data_pareto_front();
    std::string it = std::to_string(opt.iteration());
    std::string model = it + "_pareto_model.dat";
    std::string data = it + "_pareto_data.dat";
    std::ofstream pareto_model(model.c_str()),
        pareto_data(data.c_str());
    for (auto x : p_model)
      pareto_model << std::get<1>(x).transpose() << " "
                   << (std::get<1>(x).array() + std::get<2>(x)).transpose() << " "
                   << (std::get<1>(x).array() - std::get<2>(x)).transpose() << " "
                   << std::endl;
    for (auto x : p_data)
      pareto_data << std::get<1>(x).transpose() << std::endl;
  }
};



int main() {
  par::init();
  // if you want to use a standard GP & basic UCB:
  // typedef kernel_functions::MaternFiveHalfs<Params> kernel_t;
  // typedef model::GP<Params, kernel_t, mean_t> gp_t;
  // typedef acquisition_functions::UCB<Params, gp_t> ucb_t;
  //Parego<Params, model_fun<gp_t>, acq_fun<ucb_t> > opt;
  Parego<Params, stat_fun<Pareto> > opt;
  opt.optimize(mop2());

  auto p_model = opt.model_pareto_front(0, 1.0, 0.001);
  auto p_data = opt.data_pareto_front();

  std::ofstream pareto_model("mop2_pareto_model.dat"),
      pareto_data("mop2_pareto_data.dat");
  for (auto x : p_model)
    pareto_model << std::get<1>(x).transpose() << " "
                 << (std::get<1>(x).array() + std::get<2>(x)).transpose() << " "
                 << (std::get<1>(x).array() - std::get<2>(x)).transpose() << " "
                 << std::endl;
  for (auto x : p_data)
    pareto_data << std::get<1>(x).transpose() << std::endl;

  return 0;
}
