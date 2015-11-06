#ifndef LIMBO_MODEL_GP_KERNEL_MEAN_LF_OPT_HPP
#define LIMBO_MODEL_GP_KERNEL_MEAN_LF_OPT_HPP

#include <Eigen/Core>

#include <limbo/opt/rprop.hpp>
#include <limbo/opt/parallel_repeater.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            template <typename Params>
            struct KernelMeanLFOpt {
            public:
                template <typename GP>
                void operator()(GP& gp) const
                {
                    KernelMeanLFOptimization<GP> optimization(gp);
                    opt::ParallelRepeater<Params, opt::Rprop<Params>> par_rprop;
                    auto params = par_rprop(optimization);
                    gp.kernel_function().set_h_params(params.head(gp.kernel_function().h_params_size()));
                    gp.mean_function().set_h_params(params.tail(gp.mean_function().h_params_size()));
                    gp.set_lik(optimization.utility(params));
                    gp.update();
                }

            protected:
                template <typename GP>
                struct KernelMeanLFOptimization {
                public:
                    KernelMeanLFOptimization(const GP& gp) : _original_gp(gp) {}

                    double utility(const Eigen::VectorXd& params) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params.head(gp.kernel_function().h_params_size()));
                        gp.mean_function().set_h_params(params.tail(gp.mean_function().h_params_size()));

                        gp.update();

                        size_t n = gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = gp.llt().matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        // alpha = K^{-1} * this->_obs_mean;

                        // double a = this->_obs_mean.col(0).dot(this->_alpha.col(0));
                        double a = (gp.obs_mean().transpose() * gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);
                        return lik;
                    }

                    std::pair<double, Eigen::VectorXd> utility_and_grad(const Eigen::VectorXd& params) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params.head(gp.kernel_function().h_params_size()));
                        gp.mean_function().set_h_params(params.tail(gp.mean_function().h_params_size()));

                        gp.update();

                        size_t n = gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = gp.llt().matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        double a = (gp.obs_mean().transpose() * gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                        // K^{-1} using Cholesky decomposition
                        Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n, n);
                        gp.llt().matrixL().solveInPlace(K);
                        gp.llt().matrixL().transpose().solveInPlace(K);

                        // alpha * alpha.transpose() - K^{-1}
                        Eigen::MatrixXd w = gp.alpha() * gp.alpha().transpose() - K;

                        // only compute half of the matrix (symmetrical matrix)
                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(this->param_size());
                        for (size_t i = 0; i < n; ++i) {
                            for (size_t j = 0; j <= i; ++j) {
                                Eigen::VectorXd g = gp.kernel_function().grad(gp.samples()[i], gp.samples()[j]);
                                if (i == j)
                                    grad.head(gp.kernel_function().h_params_size()) += w(i, j) * g * 0.5;
                                else
                                    grad.head(gp.kernel_function().h_params_size()) += w(i, j) * g;
                            }
                        }

                        for (int i_obs = 0; i_obs < gp.dim_out(); ++i_obs)
                            for (size_t n_obs = 0; n_obs < n; n_obs++) {
                                grad.tail(gp.mean_function().h_params_size()) += gp.obs_mean().col(i_obs).transpose() * K.col(n_obs) * gp.mean_function().grad(gp.samples()[n_obs], gp).row(i_obs);
                            }

                        return std::make_pair(lik, grad);
                    }

                    size_t param_size() const
                    {
                        return this->_original_gp.kernel_function().h_params_size() + this->_original_gp.mean_function().h_params_size();
                    }

                    Eigen::VectorXd init() const
                    {
                        return (Eigen::VectorXd::Random(param_size()).array() - 1);
                    }

                protected:
                    const GP& _original_gp;
                    Eigen::VectorXd _init;
                };
            };
        }
    }
}

#endif
