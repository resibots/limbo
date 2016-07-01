#include <limbo/opt.hpp>
#include <limbo/tools.hpp>

// this short tutorial shows how to use the optimization api of limbo (opt::)
using namespace limbo;

struct ParamsGrad {
  struct opt_nloptgrad {
      BO_PARAM(int, iterations, 80);
  };
};

struct ParamsNoGrad {
  struct opt_nloptnograd {
      BO_PARAM(int, iterations, 80);
  };
};

struct ParamsCMAES {
  struct opt_cmaes : public defaults::opt_cmaes {
  };
};


// we maximize -(x_1-0.5)^2 - (x_2-0.5)^2
// the maximum is [0.5, 0.5] (f([0.5, 0.5] = 0))
opt::eval_t my_function(const Eigen::VectorXd& params, bool eval_grad = false)
{
  double v = -(params.array() - 0.5).square().sum();
  if (!eval_grad)
      return opt::no_grad(v);
  Eigen::VectorXd grad = (-2 * params).array() + 1.0;
  return {v, grad};
}


int main(int argc, char** argv)
{
#ifdef USE_NLOPT
  // the type of the optimizer (here NLOpt with the LN_LBGFGS algorithm)
  opt::NLOptGrad<ParamsGrad, nlopt::LD_LBFGS> lbfgs;
  // we start from a random point (in 2D), and the search is not bounded
  Eigen::VectorXd res_lbfgs = lbfgs(my_function, tools::random_vector(2), false);
  std::cout <<"Result with LBFGS:\t" << res_lbfgs.transpose()
            << " -> " << my_function(res_lbfgs).first << std::endl;

  // we can also use a gradient-free algorith, like DIRECT
  opt::NLOptNoGrad<ParamsNoGrad, nlopt::GN_DIRECT> direct;
  // we start from a random point (in 2D), and the search is bounded in [0,1]
  // be careful that DIRECT does not support unbounded search
  Eigen::VectorXd res_direct = direct(my_function, tools::random_vector(2), true);
  std::cout <<"Result with DIRECT:\t" << res_direct.transpose()
            << " -> " << my_function(res_direct).first << std::endl;

#endif

#ifdef USE_LIBCMAES
  // or Cmaes
  opt::Cmaes<ParamsCMAES> cmaes;
  Eigen::VectorXd res_cmaes = cmaes(my_function, tools::random_vector(2), false);
  std::cout <<"Result with CMA-ES:\t" << res_cmaes.transpose()
            << " -> " << my_function(res_cmaes).first << std::endl;
#endif

  return 0;
}
