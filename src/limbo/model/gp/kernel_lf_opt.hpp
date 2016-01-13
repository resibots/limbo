#ifndef LIMBO_MODEL_GP_KERNEL_LF_OPT_HPP
#define LIMBO_MODEL_GP_KERNEL_LF_OPT_HPP

#include <Eigen/Core>

#include <limbo/opt/rprop.hpp>
#include <limbo/opt/parallel_repeater.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            template <typename Params, typename Optimizer = opt::ParallelRepeater<Params, opt::Rprop<Params>>>
            struct KernelLFOpt {
            public:
                template <typename GP>
                void operator()(GP& gp) const
                {
                    int dim = gp.kernel_function().h_params_size();
                    KernelLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    auto params = optimizer(optimization, tools::random_vector(dim), false);
                    gp.kernel_function().set_h_params(params);
                    gp.set_lik(opt::eval(optimization, params));
                    gp.update();
                }

            protected:
                template <typename GP>
                struct KernelLFOptimization {
                public:
                    KernelLFOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.kernel_function().set_h_params(params);

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

                        if (!compute_grad)
                            return opt::no_grad(lik);

                        // K^{-1} using Cholesky decomposition
                        Eigen::MatrixXd w = Eigen::MatrixXd::Identity(n, n);
                        gp.llt().matrixL().solveInPlace(w);
                        gp.llt().matrixL().transpose().solveInPlace(w);

                        // alpha * alpha.transpose() - K^{-1}
                        w = gp.alpha() * gp.alpha().transpose() - w;

                        // only compute half of the matrix (symmetrical matrix)
                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
                        for (size_t i = 0; i < n; ++i) {
                            for (size_t j = 0; j <= i; ++j) {
                                Eigen::VectorXd g = gp.kernel_function().grad(gp.samples()[i], gp.samples()[j]);
                                if (i == j)
                                    grad += w(i, j) * g * 0.5;
                                else
                                    grad += w(i, j) * g;
                            }
                        }

                        return {lik, grad};
                    }

                protected:
                    const GP& _original_gp;
                };
            };
        }
    }
}

#endif
