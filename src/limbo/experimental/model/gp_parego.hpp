#ifndef LIMBO_MODEL_GP_PAREGO_HPP
#define LIMBO_MODEL_GP_PAREGO_HPP

#include <iostream>
#include <cassert>
#include <limits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include <limbo/model/gp/no_lf_opt.hpp>

namespace limbo {
    namespace experimental {
      namespace defaults {
          struct model_gp_parego {
              BO_PARAM(double, rho, 0.05);
          };
      }
      namespace model {

            // this is the model used in Parego (see ??)
            // - this models aggregates all the objective values with the Tchebycheff distance
            // - objectives are weighted using a random vector
            // - single model is built
            template <typename Params, typename Model>
            class GPParego : public Model {
            public:
                GPParego() {}
                GPParego(int dim_in, int dim_out) : Model(dim_in, 1), _nb_objs(dim_out) {}
                void compute(const std::vector<Eigen::VectorXd>& samples,
                    const std::vector<Eigen::VectorXd>& observations, double noise,
                    const std::vector<Eigen::VectorXd>& bl_samples = std::vector<Eigen::VectorXd>())
                {
                    auto new_observations = _scalarize_obs(observations);
                    Model::compute(samples, new_observations, noise, bl_samples);
                }

            protected:
                size_t _nb_objs;
                Eigen::VectorXd _make_v1(double x)
                {
                    Eigen::VectorXd v1(1);
                    v1 << x;
                    return v1;
                }
                std::vector<Eigen::VectorXd> _scalarize_obs(const std::vector<Eigen::VectorXd>& observations)
                {
                    Eigen::VectorXd lambda = tools::random_vector(_nb_objs);
                    double sum = lambda.sum();
                    lambda = lambda / sum;
                    // scalarize (Tchebycheff)
                    std::vector<Eigen::VectorXd> scalarized;
                    for (auto x : observations) {
                        double y = (lambda.array() * x.array()).maxCoeff();
                        double s = (lambda.array() * x.array()).sum();
                        auto v = _make_v1(y + Params::model_gp_parego::rho() * s);
                        scalarized.push_back(v);
                    }
                    return scalarized;
                }
            };
        }
    }
}

#endif
