#ifndef LIMBO_OPT_IMPL_GP_MEAN_LF_OPT_HPP
#define LIMBO_OPT_IMPL_GP_MEAN_LF_OPT_HPP

#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include <limbo/opt/rprop.hpp>

namespace limbo {
    namespace opt {
        namespace impl {
            template <typename Params, typename Model>
            struct GPMeanLFOptStruct {
            public:
                GPMeanLFOptStruct(const Model& model) : _model(model) {}

                double utility(const Eigen::VectorXd& params)
                {
                    this->_model.mean_function().set_h_params(params);

                    this->_model.update();

                    size_t n = this->_model.obs_mean().rows();

                    // --- cholesky ---
                    // see:
                    // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                    Eigen::MatrixXd l = this->_model.llt().matrixL();
                    long double det = 2 * l.diagonal().array().log().sum();

                    // alpha = K^{-1} * this->_obs_mean;

                    // double a = this->_obs_mean.col(0).dot(this->_alpha.col(0));
                    double a = (this->_model.obs_mean().transpose() * this->_model.alpha())
                                   .trace(); // generalization for multi dimensional observation
                    // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                    double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);
                    return lik;
                }

                std::pair<double, Eigen::VectorXd> utility_and_grad(const Eigen::VectorXd& params)
                {
                    this->_model.mean_function().set_h_params(params);

                    this->_model.update();

                    size_t n = this->_model.obs_mean().rows();

                    // --- cholesky ---
                    // see:
                    // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                    Eigen::MatrixXd l = this->_model.llt().matrixL();
                    long double det = 2 * l.diagonal().array().log().sum();

                    double a = (this->_model.obs_mean().transpose() * this->_model.alpha())
                                   .trace(); // generalization for multi dimensional observation
                    // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                    double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                    // K^{-1} using Cholesky decomposition
                    Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n, n);
                    this->_model.llt().matrixL().solveInPlace(K);
                    this->_model.llt().matrixL().transpose().solveInPlace(K);

                    Eigen::VectorXd grad = Eigen::VectorXd::Zero(this->param_size());
                    for (int i_obs = 0; i_obs < this->_model.dim_out(); ++i_obs)
                        for (size_t n_obs = 0; n_obs < n; n_obs++) {
                            grad.tail(this->_model.mean_function().h_params_size()) += this->_model.obs_mean().col(i_obs).transpose() * K.col(n_obs) * this->_model.mean_function().grad(this->_model.samples()[n_obs], this->_model).row(i_obs);
                        }

                    return std::make_pair(lik, grad);
                }

                size_t param_size()
                {
                    return this->_model.mean_function().h_params_size();
                }

                Eigen::VectorXd init()
                {
                    return (Eigen::VectorXd::Random(param_size()).array() - 1);
                }

            protected:
                Model _model;
                Eigen::VectorXd _init;
            };

            template <typename Params>
            struct GPMeanLFOpt {
                template <typename Opt>
                void operator()(Opt& opt, const std::vector<Eigen::VectorXd>& samples, const std::vector<Eigen::VectorXd>& observations, double noise,
                    const std::vector<Eigen::VectorXd>& bl_samples = std::vector<Eigen::VectorXd>())
                {
                    opt.compute(samples, observations, noise, bl_samples);
                    GPMeanLFOptStruct<Params, Opt> util(opt);
                    auto params = opt::par::rprop<Params>(util);
                    opt.mean_function().set_h_params(params);
                    opt.set_lik(util.utility(params));
                    opt.update();
                }
            };
        }
    }
}

#endif
