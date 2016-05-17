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

            /// this is the model used in Parego
            /// reference: Knowles, J. (2006). ParEGO: A hybrid algorithm
            /// with on-line landscape approximation for expensive multiobjective
            /// optimization problems.
            /// IEEE Transactions On Evolutionary Computation, 10(1), 50-66.
            /// Main idea:
            /// - this models aggregates all the objective values with the Tchebycheff distance
            /// - objectives are weighted using a random vector
            /// - a single model is built
            template <typename Params, typename Model>
            class GPParego : public Model {
            public:
                GPParego() {}
                GPParego(int dim_in, int dim_out) : Model(dim_in, 1), _nb_objs(dim_out) {}
                void compute(const std::vector<Eigen::VectorXd>& samples,
                    const std::vector<Eigen::VectorXd>& observations, const Eigen::VectorXd& noises,
                    const std::vector<Eigen::VectorXd>& bl_samples = std::vector<Eigen::VectorXd>())
                {
                    auto new_observations = _scalarize_obs(observations);
                    Model::compute(samples, new_observations, noises, bl_samples);
                }
                /// add sample will NOT be incremental (we call compute each time)
                void add_sample(const Eigen::VectorXd& sample, const Eigen::VectorXd& observation, double noise)
                {
                    Model::add_sample(sample, observation, noise);
                    this->compute(this->_samples,
                      _convert_observations(this->_observations), this->_noises,
                      this->_bl_samples);
                }
                void add_bl_sample(const Eigen::VectorXd& bl_sample, double noise)
                {
                  Model::add_bl_sample(bl_sample, noise);
                  this->compute(this->_samples,
                    _convert_observations(this->_observations), this->_noises,
                    this->_bl_samples);
                }


            protected:
                size_t _nb_objs;
                /// we need to do this because the GP does not store the vector of observations, but convert it to a matrix...
                std::vector<Eigen::VectorXd> _convert_observations(const Eigen::MatrixXd& m){
                  std::vector<Eigen::VectorXd> res;
                  for (int i = 0; i < m.rows(); ++i)
                    res.push_back(m.row(i));
                  return res;
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
                        auto v = tools::make_vector(y + Params::model_gp_parego::rho() * s);
                        scalarized.push_back(v);
                    }
                    return scalarized;
                }
            };
        }
    }
}

#endif
