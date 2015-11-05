#ifndef LIMBO_MODEL_GP_KERNEL_LF_OPT_HPP
#define LIMBO_MODEL_GP_KERNEL_LF_OPT_HPP

#include <Eigen/Core>

#include <limbo/opt/rprop.hpp>
#include <limbo/opt/parallel_repeater.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            template <typename Params>
            struct KernelLFOpt {
            public:
                template <typename GP>
                void operator()(GP& gp) const
                {
                    KernelLFOptimization<GP> optimization(gp);
                    opt::ParallelRepeater<Params, opt::Rprop<Params>> par_rprop;
                    auto params = par_rprop(optimization);
                    gp.kernel_function().set_h_params(params);
                    gp.set_lik(optimization.utility(params));
                    gp.update();
                }

            protected:
                template <typename GP>
                struct KernelLFOptimization {
                public:
                    KernelLFOptimization(const GP& gp) : _gp(gp) {}

                    double utility(const Eigen::VectorXd& params)
                    {
                        this->_gp.kernel_function().set_h_params(params);

                        this->_gp.update();

                        size_t n = this->_gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = this->_gp.llt().matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        // alpha = K^{-1} * this->_obs_mean;

                        // double a = this->_obs_mean.col(0).dot(this->_alpha.col(0));
                        double a = (this->_gp.obs_mean().transpose() * this->_gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);
                        return lik;
                    }

                    std::pair<double, Eigen::VectorXd> utility_and_grad(const Eigen::VectorXd& params)
                    {
                        this->_gp.kernel_function().set_h_params(params);

                        this->_gp.update();

                        size_t n = this->_gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = this->_gp.llt().matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        double a = (this->_gp.obs_mean().transpose() * this->_gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                        // K^{-1} using Cholesky decomposition
                        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);
                        this->_gp.llt().matrixL().solveInPlace(w);
                        this->_gp.llt().matrixL().transpose().solveInPlace(w);

                        // alpha * alpha.transpose() - K^{-1}
                        w = this->_gp.alpha() * this->_gp.alpha().transpose() - w;

                        // only compute half of the matrix (symmetrical matrix)
                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(this->param_size());
                        for (size_t i = 0; i < n; ++i) {
                            for (size_t j = 0; j <= i; ++j) {
                                Eigen::VectorXd g = this->_gp.kernel_function().grad(this->_gp.samples()[i], this->_gp.samples()[j]);
                                if (i == j)
                                    grad += w(i, j) * g * 0.5;
                                else
                                    grad += w(i, j) * g;
                            }
                        }

                        return std::make_pair(lik, grad);
                    }

                    size_t param_size() const
                    {
                        return this->_gp.kernel_function().h_params_size();
                    }

                    Eigen::VectorXd init() const
                    {
                        return (Eigen::VectorXd::Random(param_size()).array() - 1);
                    }

                protected:
                    GP _gp;
                    Eigen::VectorXd _init;
                };
            };
        }
    }
}

#endif
