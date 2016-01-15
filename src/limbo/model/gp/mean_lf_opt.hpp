#ifndef LIMBO_MODEL_GP_MEAN_LF_OPT_HPP
#define LIMBO_MODEL_GP_MEAN_LF_OPT_HPP

#include <Eigen/Core>

#include <limbo/opt/rprop.hpp>
#include <limbo/opt/parallel_repeater.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            template <typename Params, typename Optimizer = opt::ParallelRepeater<Params, opt::Rprop<Params>>>
            struct MeanLFOpt {
            public:
                template <typename GP>
                void operator()(GP& gp) const
                {
                    MeanLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    int dim = gp.mean_function().h_params_size();
                    auto params = optimizer(optimization, tools::random_vector(dim), false);
                    gp.mean_function().set_h_params(params);
                    gp.set_lik(opt::eval(optimization, params));
                    gp.update();
                }

            protected:
                template <typename GP>
                struct MeanLFOptimization {
                public:
                    MeanLFOptimization(const GP& gp) : _original_gp(gp) {}

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.mean_function().set_h_params(params);

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
                        Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n, n);
                        gp.llt().matrixL().solveInPlace(K);
                        gp.llt().matrixL().transpose().solveInPlace(K);

                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
                        for (int i_obs = 0; i_obs < gp.dim_out(); ++i_obs)
                            for (size_t n_obs = 0; n_obs < n; n_obs++) {
                                grad.tail(gp.mean_function().h_params_size()) += gp.obs_mean().col(i_obs).transpose() * K.col(n_obs) * gp.mean_function().grad(gp.samples()[n_obs], gp).row(i_obs);
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
