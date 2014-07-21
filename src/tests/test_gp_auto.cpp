
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE parallel

#include <boost/test/unit_test.hpp>

#include "limbo/macros.hpp"
#include "limbo/gp_auto.hpp"

using namespace limbo;

struct Params {
  struct meanconstant : public defaults::meanconstant {};
  struct gp_auto : public defaults::gp_auto {};
};

Eigen::VectorXd make_v1(double x) {
  Eigen::VectorXd v1(1);
  v1 << x;
  return v1;
}

BOOST_AUTO_TEST_CASE(test_gp_auto) {
  typedef kernel_functions::SquaredExpARD<Params> KF_t;
  typedef mean_functions::MeanConstant<Params> Mean_t;
  typedef model::GPAuto<Params, KF_t, Mean_t> GP_t;

  GP_t gp(1);
  std::vector<double> observations = {5, 10, 5};
  std::vector<Eigen::VectorXd> samples = { make_v1(1), make_v1(2), make_v1(3) };

  gp.compute(samples, observations, 0.0);

  double mu, sigma;
  std::tie(mu, sigma) = gp.query(make_v1(1));
  BOOST_CHECK_CLOSE(mu, 5, 1);
  BOOST_CHECK(sigma < 1e-5);

  std::tie(mu, sigma) = gp.query(make_v1(2));
  BOOST_CHECK_CLOSE(mu, 10, 1);
  BOOST_CHECK(sigma < 1e-5);

  std::tie(mu, sigma) = gp.query(make_v1(3));
  BOOST_CHECK_CLOSE(mu, 5, 1);
  BOOST_CHECK(sigma < 1e-5);

  std::cout << "Params=>" << gp.kernel_function().h_params().transpose() << std::endl;

  std::ofstream ofs("gp.dat");
  for (double x = 0; x < 6; x += 0.05) {
    double mu, sigma;
    std::tie(mu, sigma) = gp.query(make_v1(x));
    BOOST_CHECK(gp.mu(make_v1(x)) == mu);
    BOOST_CHECK(gp.sigma(make_v1(x)) == sigma);
    ofs << x << " " << mu << " " << mu - sigma << " " << mu + sigma << std::endl;
  }

}
