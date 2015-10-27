#ifndef GP_AUTO_MEAN_HPP_
#define GP_AUTO_MEAN_HPP_

#include <limits>
#include <cassert>
#include <tbb/parallel_reduce.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/LU>

#include <limbo/models/gp.hpp>
#include <limbo/optimizers/rprop.hpp>
#include <limbo/parallel.hpp>

namespace limbo {
    namespace defaults {
        struct gp_auto_mean {
            BO_PARAM(int, n_rprop, 100);
            BO_PARAM(int, rprop_restart, 100);
        };
    }
    namespace models {
        template <typename Params, typename KernelFunction, typename MeanFunction, typename ObsType = Eigen::VectorXd>
        class GPAutoMean : public GP<Params, KernelFunction, MeanFunction> {
        public:
            GPAutoMean() : GP<Params, KernelFunction, MeanFunction>() {}
            // TODO : init KernelFunction with dim in GP
            GPAutoMean(int dim_in, int dim_out)
                : GP<Params, KernelFunction, MeanFunction>(dim_in, dim_out) {}

            void compute(const std::vector<Eigen::VectorXd>& samples,
                const std::vector<ObsType>& observations, double noise,
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
                par::init();
                typedef std::pair<Eigen::VectorXd, double> pair_t;
                auto body = [=](int i) {
                    // clang-format off
                    // we need a copy because each thread should touch a copy of the GP!
                    auto gp = *this;
                    Eigen::VectorXd v = rprop::optimize([&](const Eigen::VectorXd & v) {
                      return gp.log_likelihood(v);
                    },
                    [&](const Eigen::VectorXd & v) {
                      return gp.log_likelihood_grad(v, false);
                    },
          	        this->kernel_function().h_params_size()+this->_mean_function.h_params_size(), Params::gp_auto_mean::n_rprop());

                    double lik = gp.log_likelihood(v);//this->log_likelihood(v);
                    return std::make_pair(v, lik);
                    // clang-format on
                };
                auto comp = [](const pair_t& v1, const pair_t& v2) {
                    // clang-format off
                    return v1.second > v2.second;
                    // clang-format on
                };
                pair_t init(Eigen::VectorXd::Zero(1), -std::numeric_limits<float>::max());
                auto m = par::max(init, Params::gp_auto_mean::rprop_restart(), body, comp);
                std::cout << "likelihood:" << m.second << std::endl;

                this->_kernel_function.set_h_params(
                    m.first.head(this->_kernel_function.h_params_size()));
                this->_mean_function.set_h_params(
                    m.first.tail(this->_mean_function.h_params_size()));
                this->_lik = m.second;
            }
        };
    }
}
#endif
