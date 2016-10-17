//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_MODEL_GP_MEAN_LF_OPT_HPP
#define LIMBO_MODEL_GP_MEAN_LF_OPT_HPP

#include <limbo/model/gp/hp_opt.hpp>
#include <limbo/tools/random_generator.hpp>

namespace limbo {
    namespace model {
        namespace gp {
            ///@ingroup model_opt
            ///optimize the likelihood of the mean only (try to align the mean function)
            template <typename Params, typename Optimizer = opt::ParallelRepeater<Params, opt::Rprop<Params>>>
            struct MeanLFOpt : public HPOpt<Params, Optimizer> {
            public:
                template <typename GP>
                void operator()(GP& gp)
                {
                    this->_called = true;
                    MeanLFOptimization<GP> optimization(gp);
                    Optimizer optimizer;
                    auto params = optimizer(optimization, gp.mean_function().h_params(), false);
                    gp.mean_function().set_h_params(params);
                    gp.set_lik(opt::eval(optimization, params));
                    gp.recompute_mean_internal();
                }

            protected:
                template <typename GP>
                struct MeanLFOptimization {
                public:
                    MeanLFOptimization(const GP& gp) : _original_gp(gp)
                    {
                        size_t n = gp.obs_mean().rows();

                        // K^{-1} using Cholesky decomposition
                        _K = Eigen::MatrixXd::Identity(n, n);
                        gp.matrixL().template triangularView<Eigen::Lower>().solveInPlace(_K);
                        gp.matrixL().template triangularView<Eigen::Lower>().transpose().solveInPlace(_K);
                    }

                    opt::eval_t operator()(const Eigen::VectorXd& params, bool compute_grad) const
                    {
                        GP gp(this->_original_gp);
                        gp.mean_function().set_h_params(params);

                        gp.recompute_mean_internal();

                        size_t n = gp.obs_mean().rows();

                        // --- cholesky ---
                        // see:
                        // http://xcorr.net/2008/06/11/log-determinant-of-positive-definite-matrices-in-matlab/
                        Eigen::MatrixXd l = gp.matrixL();
                        long double det = 2 * l.diagonal().array().log().sum();

                        double a = (gp.obs_mean().transpose() * gp.alpha())
                                       .trace(); // generalization for multi dimensional observation
                        // std::cout<<" a: "<<a <<" det: "<< det<<std::endl;
                        double lik = -0.5 * a - 0.5 * det - 0.5 * n * log(2 * M_PI);

                        if (!compute_grad)
                            return opt::no_grad(lik);

                        Eigen::VectorXd grad = Eigen::VectorXd::Zero(params.size());
                        for (int i_obs = 0; i_obs < gp.dim_out(); ++i_obs)
                            for (size_t n_obs = 0; n_obs < n; n_obs++) {
                                grad.tail(gp.mean_function().h_params_size()) += gp.obs_mean().col(i_obs).transpose() * _K.col(n_obs) * gp.mean_function().grad(gp.samples()[n_obs], gp).row(i_obs);
                            }

                        return {lik, grad};
                    }

                protected:
                    const GP& _original_gp;
                    Eigen::MatrixXd _K;
                };
            };
        }
    }
}

#endif
