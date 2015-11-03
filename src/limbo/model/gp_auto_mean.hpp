#ifndef LIMBO_MODEL_GP_AUTO_MEAN_HPP
#define LIMBO_MODEL_GP_AUTO_MEAN_HPP

#include <iostream>
#include <cassert>
#include <limits>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include <limbo/tools/macros.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/opt/rprop.hpp>
#include <limbo/tools/parallel.hpp>

namespace limbo {
    namespace defaults {
        struct gp_auto_mean {
            BO_PARAM(int, n_rprop, 100);
            BO_PARAM(int, rprop_restart, 100);
        };
    }

    template <typename Params, typename Model>
    struct LikelihoodKernelMean {
        LikelihoodKernelMean(Model& model) : _model(model) {}

        std::pair<double, Eigen::VectorXd> utility_and_grad(const Eigen::VectorXd& params)
        {
            return std::make_pair(_model.log_likelihood(params, true), _model.log_likelihood_grad(params));
        }

        double utility(const Eigen::VectorXd& params)
        {
            return _model.log_likelihood(params, true);
        }

        size_t param_size()
        {
            return _model.kernel_function().h_params_size()+_model.mean_function().h_params_size();
        }

    private:
        Model _model;
    };

    namespace model {
        template <typename Params, typename KernelFunction, typename MeanFunction>
        class GPAutoMean : public GP<Params, KernelFunction, MeanFunction> {
        public:
            GPAutoMean() : GP<Params, KernelFunction, MeanFunction>() {}
            // TODO : init KernelFunction with dim in GP
            GPAutoMean(int dim_in, int dim_out)
                : GP<Params, KernelFunction, MeanFunction>(dim_in, dim_out) {}

            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<Eigen::VectorXd>& observations, double noise,
                const std::vector<Eigen::VectorXd>& bl_samples = std::vector<Eigen::VectorXd>())
            {

                GP<Params, KernelFunction, MeanFunction>::compute(samples, observations,
                    noise, bl_samples);
                _optimize_likelihood();
                //	  std::cout<<"kernel params:
                //"<<this->_kernel_function.h_params().transpose()<<std::endl;
                //      std::cout<<"mean params:
                //      "<<this->_mean_function.h_params().transpose()<<std::endl;

                this->_compute_obs_mean(); // ORDER MATTERS
                this->_compute_kernel();
            }

            Eigen::VectorXd check_inverse()
            {

                return this->_kernel * this->_alpha.col(0) - this->_obs_mean;
            }

            // see Rasmussen and Williams, 2006 (p. 113)
            virtual double log_likelihood(const Eigen::VectorXd& h_params, bool update_kernel = true)
            {
                this->_kernel_function.set_h_params(
                    h_params.head(this->_kernel_function.h_params_size()));
                this->_mean_function.set_h_params(
                    h_params.tail(this->_mean_function.h_params_size()));
                if (update_kernel) {
                    this->_compute_obs_mean();
                    this->_compute_kernel();
                }
                size_t n = this->_obs_mean.rows();

                // --- cholesky ---
                // see:
                // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                Eigen::MatrixXd l = this->_llt.matrixL();
                long double det = 2 * l.diagonal().array().log().sum();

                // alpha = K^{-1} * this->_obs_mean;

                //        double a = this->_obs_mean.col(0).dot(this->_alpha.col(0));
                double a = (this->_obs_mean.transpose() * this->_alpha)
                               .trace(); // generalization for multi dimensional observation
                // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);
                // std::cout<<h_params.transpose()<<"   "<<lik<<std::endl;
                return lik;
            }

            // see Rasmussen and Williams, 2006 (p. 114)
            virtual Eigen::VectorXd log_likelihood_grad(const Eigen::VectorXd& h_params, bool update_kernel = true)
            {
                this->_kernel_function.set_h_params(
                    h_params.head(this->_kernel_function.h_params_size()));
                this->_mean_function.set_h_params(
                    h_params.tail(this->_mean_function.h_params_size()));

                if (update_kernel) {
                    this->_compute_obs_mean(); // ORDER MATTERS
                    this->_compute_kernel();
                }
                size_t n = this->_observations.rows();

                // K^{-1} using Cholesky decomposition
                Eigen::MatrixXd K = Eigen::MatrixXd::Identity(n, n);
                this->_llt.matrixL().solveInPlace(K);
                this->_llt.matrixL().transpose().solveInPlace(K);

                // alpha * alpha.transpose() - K^{-1}
                Eigen::MatrixXd w = this->_alpha * this->_alpha.transpose() - K;

                // only compute half of the matrix (symmetrical matrix)
                Eigen::VectorXd grad = Eigen::VectorXd::Zero(this->_kernel_function.h_params_size() + this->_mean_function.h_params_size());
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        Eigen::VectorXd g = this->_kernel_function.grad(this->_samples[i], this->_samples[j]);
                        if (i == j)
                            grad.head(this->_kernel_function.h_params_size()) += w(i, j) * g * 0.5;
                        else
                            grad.head(this->_kernel_function.h_params_size()) += w(i, j) * g;
                    }
                }
                for (int i_obs = 0; i_obs < this->_dim_out; ++i_obs)
                    for (size_t n_obs = 0; n_obs < n; n_obs++) {
                        // std::cout<<"mean grad for
                        // "<<this->_samples[n_obs].transpose()<<std::endl<<this->_mean_function.grad(this->_samples[n_obs],*this)<<std::endl;
                        grad.tail(this->_mean_function.h_params_size()) += this->_obs_mean.col(i_obs).transpose() * K.col(n_obs) * this->_mean_function.grad(this->_samples[n_obs], *this).row(i_obs);
                    }
                return grad;
            }

            float get_lik() const { return _lik; }

        protected:
            float _lik;

            virtual void _optimize_likelihood()
            {
                LikelihoodKernelMean<Params, GPAutoMean<Params, KernelFunction, MeanFunction>> kk(*this);
                auto bp = opt::par::rprop<Params>(kk);
                double lik = kk.utility(bp);

                std::cout << "likelihood:" << lik << std::endl;
                this->_kernel_function.set_h_params(
                    bp.head(this->_kernel_function.h_params_size()));
                this->_mean_function.set_h_params(
                    bp.tail(this->_mean_function.h_params_size()));
                this->_lik = lik;
            }
        };
    }
}

#endif
