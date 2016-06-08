#include <limbo/tools/macros.hpp>
#include <limbo/kernel/exp.hpp>
#include <limbo/kernel/squared_exp_ard.hpp>
#include <limbo/mean/data.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/tools.hpp>
#include <fstream>

// this tutorials shows how to use a Gaussian process for regression

using namespace limbo;

struct Params {
  struct kernel_exp {
    BO_PARAM(double, sigma, 0.15);
  };
  struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
  };
};

int main(int argc, char **argv)
{

  // our data (1-D inputs, 1-D outputs)
  std::vector<Eigen::VectorXd> samples = {
    tools::make_vector(0),
    tools::make_vector(0.25),
    tools::make_vector(0.5),
    tools::make_vector(1.0)};
  std::vector<Eigen::VectorXd> observations = {
    tools::make_vector(-1.0),
    tools::make_vector(2.0),
    tools::make_vector(1.0),
    tools::make_vector(3.0) };

  // the type of the GP
  typedef kernel::Exp<Params> Kernel_t;
  typedef mean::Data<Params> Mean_t;
  typedef model::GP<Params, Kernel_t, Mean_t> GP_t;

  // 1-D inputs, 1-D outputs
  GP_t gp(1, 1);

  // noise is the same for all the samples (0.05)
  Eigen::VectorXd noises = Eigen::VectorXd::Ones(samples.size()) * 0.01;

  // compute the GP
  gp.compute(samples, observations, noises);

  // write the predicted data in a file (e.g. to be plotted)
  std::ofstream ofs("gp.dat");
  for (int i = 0; i < 100; ++i) {
    Eigen::VectorXd v = tools::make_vector(i / 100.0);
    Eigen::VectorXd mu; double sigma;
    std::tie(mu, sigma) = gp.query(v);
    // an alternative (slower) is to query mu and sigma separately:
    //  double mu = gp.mu(v)[0]; // mu() returns a 1-D vector
    //  double s2 = gp.sigma(v);
    ofs << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
  }

  // an alternative is to optimize the hyper-paramerers
  // in that case, we need a kernel with hyper-parameters that are designed to be optimized
  typedef kernel::SquaredExpARD<Params> Kernel2_t;
  typedef mean::Data<Params> Mean_t;
  typedef model::GP<Params, Kernel2_t, Mean_t> GP2_t;

  GP2_t gp_ard(1, 1);
  // do not forget to call the optimization!
  gp_ard.optimize_hyperparams();
  gp_ard.compute(samples, observations, noises);


  // write the predicted data in a file (e.g. to be plotted)
  std::ofstream ofs_ard("gp_ard.dat");
  for (int i = 0; i < 100; ++i) {
    Eigen::VectorXd v = tools::make_vector(i / 100.0);
    Eigen::VectorXd mu; double sigma;
    std::tie(mu, sigma) = gp_ard.query(v);
    ofs_ard << v.transpose() << " " << mu[0] << " " << sqrt(sigma) << std::endl;
  }

  // write the data to a file (useful for plotting)
  std::ofstream ofs_data("data.dat");
  for (int i = 0; i < samples.size(); ++i)
    ofs_data << samples[i].transpose() << " " << observations[i].transpose() << std::endl;
  return 0;
}
